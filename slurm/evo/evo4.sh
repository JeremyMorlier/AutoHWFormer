#!/bin/bash
#SBATCH --account             sxq@a100
#SBATCH --constraint          a100
#SBATCH --cpus-per-task       10
#SBATCH --error               log/%x/%j/errors.err
#SBATCH --gres                gpu:4
#SBATCH --hint                nomultithread
#SBATCH --job-name            HWSuper_Test
#SBATCH --nodes               1
#SBATCH --ntasks              4
#SBATCH --output              log/%x/%j/logs.out
#SBATCH --qos                 qos_gpu_a100-t3 
#SBATCH --time                20:00:00
#SBATCH --signal              USR1@40

# This experiment is the Autoformer SuperNet Tiny with an evolutionary search of 5M to 7M
module purge
conda deactivate

module load arch/a100
source .venv/bin/activate

srun python3 evolution_subnet.py --data-path $DSDIR/imagenet --output_dir results/evoT5m7m --gp --change_qk --relative_position --dist-eval --cfg config/supernet-T.yaml --resume checkpoints/supernet-tiny.pth --min-param-limits 5 --param-limits 7 --data-path $DSDIR/imagenet --evo-data-path $SCRATCH/EvoImageNet/ --data-set EVO_IMNET