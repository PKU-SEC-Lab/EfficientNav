## EfficientNav: On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval
This is also the official code repository for the paper [EfficientNav](https://arxiv.org/abs/2510.18546).

EfficientNav is a novel framework that enables efficient on-device Object Goal Navigation (ObjNav) using smaller language models. Developing LLM-based navigation system on local device is challenging, due to the limited model capacity of smaller LLM planner for understanding complex navigation maps.
At the same time, the long prompt introduced by the navigation map description will cause high planning latency on local devices. 
This project tackles the critical challenges of deploying LLM-based navigation agents on local devices by efficient navigation map caching and retrieval.

## Key Features 🚀 
- **Semantics-Aware Memory Retrieval**: Prunes redundant information in navigation maps to enhance smaller LLMs' environment understanding

- **Discrete Memory Caching**: Efficiently saves and reuses KV cache to reduce planning latency

- **Attention-Based Memory Clustering**: Recovers memory interactions for better model performance

- On the HM3D dataset, EfficientNav significantly reduces KV-cache recomputation and memory usage while improving navigation success rates—**even outperforming GPT-4-based planners**

## Installation
Assuming you have conda installed, let's prepare a conda env:
```
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```
Install required packages:
```
pip install -r requirements.txt
```
Install habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
conda install habitat-sim headless -c conda-forge -c aihabitat
```
Install habitat-lab:
```
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```
Install groundingdino:
```
export CUDA_HOME=/path/to/cuda
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```
Download CLIP checkpoint from https://huggingface.co/openai/clip-vit-base-patch32/tree/main.

Download LLaVA-34b model checkpoint from https://huggingface.co/llava-hf/llava-v1.6-34b-hf/tree/main.

Download habitat challenge scenes into `./data` from https://matterport.com/partners/meta.

## Running
```
python efficientnav.py 
```

## Citation
```
@article{yang2025efficientnav,
  title={EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval},
  author={Yang, Zebin and Zheng, Sunjian and Xie, Tong and Xu, Tianshi and Yu, Bo and Wang, Fan and Tang, Jie and Liu, Shaoshan and Li, Meng},
  journal={arXiv preprint arXiv:2510.18546},
  year={2025}
}
```

