
#SBATCH --account             sxq@v100
#SBATCH --constraint          v100-32g
#SBATCH --cpus-per-task       10
#SBATCH --error               log/%x/%j/errors.err
#SBATCH --gres                gpu:4
#SBATCH --hint                nomultithread
#SBATCH --job-name            3_ResNet
#SBATCH --nodes               1
#SBATCH --ntasks              4
#SBATCH --output              log/%x/%j/logs.out
#SBATCH --qos                 qos_gpu-t3
#SBATCH --time                20:00:00
#SBATCH --signal              USR1@40


module purge
conda deactivate

source .venv/bin/activate

srun python3 train_supernet.py --cfg config/supernet-B.yaml --output_dir results --logger txt --data-path $DSDIR/imagenet 