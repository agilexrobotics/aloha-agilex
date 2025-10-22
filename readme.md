
# ALOHA
Referencing https://github.com/MarkFzp/mobile-aloha.git, For more details, please refer to this link.
## Installation
ubuntu22.04 + cuda12.8
```bash
# Clone this repo
git clone https://github.com/agilexrobotics/aloha-agilex.git aloha
cd aloha

# Create a Conda environment
conda create -n aloha python=3.10.0
conda activate aloha
pip install -r requirements.txt
```

## Train
```bash
conda activate aloha && cd ~/aloha/act && python aloha_train.py --dataset_dir /home/agilex/data --ckpt_dir /home/agilex/checkpoint_aloha
```

## Inference
```bash
conda activate aloha && cd ~/aloha/act && python aloha_inference-ros2.py --ckpt_dir /home/agilex/checkpoint_aloha
```