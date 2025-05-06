#!/bin/bash

###
# This script must be launched with sbatch <newjob.sh>
###

# Submission script for Manneback
#SBATCH --job-name=anonymize
#SBATCH --time=05:00:00 # hh:mm:ss
#
## MORE INFO HERE:
## https://www.cism.ucl.ac.be/doc/_contents/Computing/index.html?highlight=gpu#manneback-cluster
#SBATCH --ntasks=1
#SBATCH --gres="gpu:TeslaA10:1"
#SBATCH --mem-per-cpu=102400 # 100GB
#SBATCH --partition=gpu
#SBATCH --mail-user=xavier.gillard@uclouvain.be
#SBATCH --mail-type=ALL
#SBATCH --output=out.txt

module --force purge
module --ignore_cache load              \
	releases/2023a                  \
	poetry/1.5.1-GCCcore-12.3.0     \
	SciPy-bundle/2023.07-gfbf-2023a \
	PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 

. .venv/bin/activate
poetry config virtualenvs.options.system-site-packages true --local
srun poetry run python -m finetune_gemma3.anonymize
#srun poetry run python -m finetune_gemma3.train

