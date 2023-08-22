#!/bin/bash

#SBATCH --job-name=retrocode

#SBATCH --output=./logfiles/logfile_conala_seq_finetuneemb.out

#SBATCH --error=./logfiles/logfile_conala_seq__finetuneemb.err

#SBATCH --time=08:30:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=40

#SBATCH --hint=nomultithread


module purge
module load anaconda-py3/2019.03
conda activate retrocode
set -x
nvidia-smi
# This will create a config file on your server


bash dataset/nl/seq2seq/en2fr/prepare-wmt14en2fr.sh
bash dataset/nl/seq2seq/en2de/prepare-wmt14en2de.sh

bash dataset/nl/lm/en2fr/prepare-wmt14en2fr.sh
bash dataset/nl/lm/en2de/prepare-wmt14en2de.sh