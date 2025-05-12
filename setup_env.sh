# Install miniconda for the system
# From https://www.anaconda.com/docs/getting-started/miniconda/install#linux
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

# Create conda environment
conda env create --file environment.yml

# Updates conda
conda update -n base -c defaults conda

# Update environment
conda activate musica-env

# Login manually
wandb login
huggingface-cli login