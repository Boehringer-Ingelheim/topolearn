from argparse import ArgumentParser, Namespace
from datetime import date
import logging
from os import PathLike
from pathlib import Path
from random import choices
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tdc.generation import MolGen
from tdc.metadata import single_molecule_dataset_names as DATASETS
import torch
from torch import nn
import torch.utils.data
import torchdrug.data

from ae_utils.modules import RnnDecoder, RnnEncoder
from ae_utils.char import (
    LitCVAE,
    Tokenizer,
    UnsupervisedDataset,
    CachedUnsupervisedDataset,
    SemisupervisedDataset,
)
from rogi_xd.models.gin import LitAttrMaskGIN, CustomDataset

from rogi_xd.utils import CACHE_DIR
from rogi_xd.cli.utils import NOW
from rogi_xd.cli.utils.args import ModelType, bounded, fuzzy_lookup
from rogi_xd.cli.utils.command import Subcommand

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


class TrainSubcommand(Subcommand):
    COMMAND = "train"
    HELP = "pretrain a VAE or GIN model via unsupervised learning"

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("model", type=ModelType.get, choices=list(ModelType))
        parser.add_argument("-o", "--output", help="where to save")
        xor_group = parser.add_mutually_exclusive_group(required=True)
        xor_group.add_argument(
            "-i",
            "--input",
            type=Path,
            help="a plaintext file containing one SMILES string per line. Mutually exclusive with the '--dataset' argument.",
        )
        xor_group.add_argument(
            "-d",
            "--dataset",
            type=fuzzy_lookup(DATASETS),
            choices=DATASETS,
            help="the TDC molecule generation dataset to train on. For more details, see https://tdcommons.ai/generation_tasks/molgen. Mutually exclusive with the '--input' argument",
        )
        parser.add_argument(
            "-N",
            type=bounded(lo=1)(int),
            help="the number of SMILES strings to subsample. Must be >= 1",
        )
        parser.add_argument(
            "-c",
            "--num-workers",
            type=int,
            default=-1,
            help="the number of workers to use for data loading. 0 means no parallel dataloading and -1 means to cache the featurizations. NOTE: caching may not be possible for some datasets due to memory constraints on your machine.",
        )
        parser.add_argument(
            "-g",
            "--gpus",
            type=int,
            help="the number of GPUs to use (if any). If unspecified, will use GPU if available",
        )
        parser.add_argument(
            "--chkpt",
            help="the path of a checkpoint file from a previous run from which to resume training",
        )

        return parser

    @staticmethod
    def func(args: Namespace):
        if args.input:
            smis = args.input.read_text().splitlines()
        else:
            smis = MolGen(args.dataset, CACHE_DIR).get_data().smiles.tolist()

        if args.N:
            smis = choices(smis, k=args.N)

        if args.gpus is None:
            logger.debug("GPU unspecified... Will use GPU if available")
            args.gpus = 1 if torch.cuda.is_available() else 0

        if len(smis) == 0:
            raise ValueError("No smiles strings were supplied!")

        if args.model == ModelType.GIN:
            func = TrainSubcommand.train_gin
        elif args.model == ModelType.VAE:
            func = TrainSubcommand.train_vae
        else:
            raise RuntimeError("Help! I've fallen and I can't get up! << CALL LIFEALERT >>")

        model, output_dir = func(
            smis,
            args.dataset or args.input.stem,
            args.output,
            args.num_workers,
            args.gpus,
            args.chkpt,
        )

        if next(output_dir.iterdir(), None) is not None:
            new_dir = output_dir.with_name(f"{output_dir.name}.{NOW}")
            model.save(new_dir)
            logger.info(
                f"{output_dir} is not empty! Saved {args.model} model to {new_dir} instead..."
            )
        else:
            model.save(output_dir)
            logger.info(f"Saved {args.model} model to {new_dir}")

    @staticmethod
    def train_gin(
        smis: list[str],
        dataset_name: str,
        output_dir: Optional[PathLike],
        num_workers: int = 0,
        gpus: Optional[int] = None,
        chkpt: Optional[PathLike] = None,
    ) -> tuple[pl.LightningModule, Path]:
        MODEL_NAME = "gin"
        TODAY = date.today().isoformat()

        dataset = CustomDataset()
        dataset.load_smiles(smis, {}, lazy=True, atom_feature="pretrain", bond_feature="pretrain")

        n_train = int(0.8 * len(dataset))
        n_val = len(dataset) - n_train
        train_dset, val_dset = torch.utils.data.random_split(dataset, [n_train, n_val])

        model = LitAttrMaskGIN(dataset.node_feature_dim, dataset.edge_feature_dim)
        checkpoint = ModelCheckpoint(
            dirpath=f"chkpts/{MODEL_NAME}/{dataset_name}/{TODAY}",
            filename="step={step:0.2e}-loss={val/loss:0.2f}-acc={val/accuracy:.2f}",
            monitor="val/loss",
            auto_insert_metric_name=False,
            save_last=True,
        )
        early_stopping = EarlyStopping("val/loss")

        trainer = pl.Trainer(
            WandbLogger(project=f"{MODEL_NAME}-{dataset_name}"),
            callbacks=[checkpoint, early_stopping],
            accelerator="gpu" if gpus else "cpu",
            devices=gpus or 1,
            check_val_every_n_epoch=3,
            max_epochs=100,
        )

        batch_size = 256
        train_loader = torchdrug.data.DataLoader(train_dset, batch_size, num_workers=num_workers)
        val_loader = torchdrug.data.DataLoader(val_dset, batch_size, num_workers=num_workers)

        if chkpt:
            logger.info(f"Resuming training from checkpoint '{chkpt}'")

        trainer.fit(model, train_loader, val_loader, ckpt_path=chkpt)
        output_dir = output_dir or f"models/{MODEL_NAME}/{dataset_name}"

        return model, Path(output_dir)

    @staticmethod
    def train_vae(
        smis: list[str],
        dataset_name: str,
        output_dir: Optional[PathLike],
        num_workers: int = 0,
        gpus: Optional[int] = None,
        chkpt: Optional[PathLike] = None,
    ):
        MODEL_NAME = "vae"
        TODAY = date.today().isoformat()

        tokenizer = Tokenizer.smiles_tokenizer()
        embedding = nn.Embedding(len(tokenizer), 64, tokenizer.PAD)
        encoder = RnnEncoder(embedding)
        decoder = RnnDecoder(tokenizer.SOS, tokenizer.EOS, embedding)
        model = LitCVAE(tokenizer, encoder, decoder)

        cache = num_workers == -1
        dset_cls = CachedUnsupervisedDataset if cache else UnsupervisedDataset
        dset = dset_cls(smis, tokenizer)
        dset = SemisupervisedDataset(dset, None)

        n_train = int(0.8 * len(dset))
        n_val = len(dset) - n_train
        train_set, val_set = torch.utils.data.random_split(dset, [n_train, n_val])

        logger = WandbLogger(project=f"{MODEL_NAME}-{dataset_name}")
        checkpoint = ModelCheckpoint(
            dirpath=f"chkpts/{MODEL_NAME}/{dataset_name}/{TODAY}",
            filename="step={step:0.2e}-loss={val/loss:0.2f}-acc={val/accuracy:.2f}",
            monitor="val/loss",
            auto_insert_metric_name=False,
            save_last=True,
        )
        early_stopping = EarlyStopping("val/loss", patience=5)
        trainer = pl.Trainer(
            logger,
            callbacks=[checkpoint, early_stopping],
            accelerator="gpu" if gpus else "cpu",
            devices=gpus or 1,
            check_val_every_n_epoch=1,
            max_epochs=100,
        )

        batch_size = 256
        num_workers = 0 if cache else num_workers
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size, num_workers=num_workers, collate_fn=dset.collate_fn
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size, num_workers=num_workers, collate_fn=dset.collate_fn
        )

        if chkpt:
            logger.info(f"Resuming training from checkpoint '{chkpt}'")

        trainer.fit(model, train_loader, val_loader, ckpt_path=chkpt)
        output_dir = output_dir or f"models/{MODEL_NAME}/{dataset_name}"

        return model, Path(output_dir)
