import yaml
import os


def generate_script(checkpoint_dir):
    return f"""#!/bin/bash

#SBATCH -A kreshuk                              
#SBATCH -N 1				            
#SBATCH -n 2				            
#SBATCH --mem 32G			            
#SBATCH -t 48:00:00                     
#SBATCH -o {checkpoint_dir}/train.log			        
#SBATCH -e {checkpoint_dir}/error.log
#SBATCH --mail-type=FAIL,BEGIN,END		    
#SBATCH --mail-user=adrian.wolny@embl.de
#SBATCH -p gpu				            
#SBATCH -C gpu=1080Ti			        
#SBATCH --gres=gpu:1	

module load cuDNN

export PYTHONPATH="/g/kreshuk/wolny/workspace/pytorch-3dunet:$PYTHONPATH"

/g/kreshuk/wolny/workspace/pytorch-3dunet/train.py --config {checkpoint_dir}/config_train.yml
"""


def _get_config_paths(root_dir):
    config_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            config_file = os.path.join(root, f)
            if config_file.endswith('.yml'):
                config_files.append(config_file)
    return config_files


if __name__ == "__main__":
    for i, config_file in enumerate(_get_config_paths('/home/adrian/workspace/for-pytorch-3dunet/FOR_paper_ovules')):
        print('Processing', config_file)
        config = yaml.load(open(config_file, 'r'))
        checkpoint_dir = config['trainer']['checkpoint_dir']
        slurm_script = generate_script(checkpoint_dir)
        with open(f'ovules_train_{i}.sh', 'w') as f:
            f.write(slurm_script)
