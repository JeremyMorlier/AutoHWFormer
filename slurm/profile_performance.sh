#!/bin/bash
#SBATCH --account             sxq@a100
#SBATCH --constraint          a100
#SBATCH --cpus-per-task       10
#SBATCH --error               log/%x/%j/errors.err
#SBATCH --gres                gpu:4
#SBATCH --hint                nomultithread
#SBATCH --job-name            AHF_Profile
#SBATCH --nodes               1
#SBATCH --ntasks              4
#SBATCH --output              log/%x/%j/logs.out
#SBATCH --qos                 qos_gpu_a100-t3 
#SBATCH --time                20:00:00
#SBATCH --signal              USR1@40


module purge
conda deactivate

source .venv/bin/activate

srun python3 profile_performance.py --cfg config/supernet-B1.yaml --output_dir results --logger txt --data-path $DSDIR/imagenet --gp --change_qk --relative_position --mode super --dist-eval  --epochs 500 --warmup-epochs 20 --batch-size 512