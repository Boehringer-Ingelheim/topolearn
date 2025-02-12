#!/bin/bash

# Activate environment
. .phd/bin/activate

# List of dataset names
datasets=("ADRA1A" "ALOX5AP" "ATR" "JAK1" "JAK2" "MUSC1" "MUSC2" "KOR" "LIPO" "HLMC" "SOL" "DPP4")

# Splitting type (must match with splitting selected in config.py)
type="random"

for name in "${datasets[@]}"; do
  sbatch_script=$(mktemp)
  cat <<EOT > $sbatch_script
#!/bin/bash
#SBATCH --job-name=analysis_${type}_$name
#SBATCH --output=analysis_${type}_$name.txt
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=8GB
#SBATCH --ntasks=1

export TOKENIZERS_PARALLELISM=true

python -m topolearn.analysis -d $name
EOT

  sbatch $sbatch_script
  rm $sbatch_script
done