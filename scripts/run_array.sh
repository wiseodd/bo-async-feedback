#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --qos=cpu_qos
#SBATCH --time=36:00:00
#SBATCH --array=6,7,8,9,10
#SBATCH --job-name=chem_${1}_${2}_${3}_${4}
#SBATCH --output=run_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akristiadi@vectorinstitute.ai

python chem_bo.py --problem=${1} --method=${2} ${3} --expert-prob=${4} --randseed=\$SLURM_ARRAY_TASK_ID
EOT
