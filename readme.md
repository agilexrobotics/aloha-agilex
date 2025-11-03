
# ALOHA
Referencing https://github.com/MarkFzp/mobile-aloha.git, For more details, please refer to this link.
Data collection method reference: https://github.com/agilexrobotics/data_tools, use hdf5 to training.

## Installation
ubuntu22.04 + cuda12.8 + ros2
```bash
# Clone this repo
git clone https://github.com/agilexrobotics/aloha-agilex.git aloha
cd aloha

# Create a Conda environment
conda create -n aloha python=3.10.0
conda activate aloha
pip install -r requirements.txt
```

### cobot_magic
## cobot_magic train
```bash
conda activate aloha && cd ~/aloha/act && python aloha_train.py --dataset_dir /home/agilex/data --ckpt_dir /home/agilex/checkpoint_aloha
```

## cobot_magic inference
Enable and open the piper arm ROS node, then
```bash
conda activate aloha && cd ~/aloha/act && python aloha_inference-ros2.py --ckpt_dir /home/agilex/checkpoint_aloha
```

### single pika
## single pika train
```bash
conda activate aloha && cd ~/aloha/act && python single_pika_train.py --dataset_dir /home/agilex/data --ckpt_dir /home/agilex/checkpoint_pika
```

## single pika inference
Enable and open the piper arm ROS node, then
```bash
conda activate pika && cd ~/pika_ros/scripts && bash start_single_gripper.bash

conda activate pika && cd ~/pika_ros && source install/setup.bash && cd ~/aloha-devel/aloha/scripts && python piper_FK-ros2.py 

conda activate pika && cd ~/pika_ros && source install/setup.bash && cd ~/aloha-devel/aloha/scripts && python piper_IK-ros2.py 

conda activate aloha && cd ~/aloha/act && python single_pika_inference-ros2.py --ckpt_dir /home/agilex/checkpoint_aloha
```


### multi pika
## multi pika train
```bash
conda activate aloha && cd ~/aloha/act && python multi_pika_train.py --dataset_dir /home/agilex/data --ckpt_dir /home/agilex/checkpoint_aloha
```

## multi pika inference
Modify multi_pika_inference.py, change "[0, -0.5, 0, 0, 0, 0]" to your right arm base in left arm base axis.
```bash
parser.add_argument('--arm_base_link_in_world', action='store', type=float, help='arm_base_link_in_world', default=[[0, 0, 0, 0, 0, 0], [0, -0.5, 0, 0, 0, 0]], required=False)
```
With the x-axis pointing forward, the y-axis pointing left, and the z-axis pointing up, for example, [0, -0.5, 0, 0, 0, 0] indicates that the right arm is installed 0.5 meters to the right of the left arm.

Enable and open the piper arm ROS node, then
```bash

conda activate pika && cd ~/pika_ros/scripts && bash start_multi_gripper.bash

conda activate pika && cd ~/pika_ros && source install/setup.bash && cd ~/aloha-devel/aloha/scripts && python piper_FK-ros2.py --index_name _l

conda activate pika && cd ~/pika_ros && source install/setup.bash && cd ~/aloha-devel/aloha/scripts && python piper_FK-ros2.py --index_name _r

conda activate pika && cd ~/pika_ros && source install/setup.bash && cd ~/aloha-devel/aloha/scripts && python piper_IK-ros2.py --index_name _l

conda activate pika && cd ~/pika_ros && source install/setup.bash && cd ~/aloha-devel/aloha/scripts && python piper_IK-ros2.py --index_name _r

conda activate aloha && cd ~/aloha/act && python multi_pika_inference-ros2.py --ckpt_dir /home/agilex/checkpoint_aloha
```

