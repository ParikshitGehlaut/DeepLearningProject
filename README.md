# CS428 Deep Learning Project

# Introduction
In our project, We succssfully trained and inference LLaST speech to text llm for German to English translations. This repository provides the methodologies and resources for advancing speech translation through innovative LLM-driven techniques.

# About Model
We address the limitations of end-to-end speech translation~(E2E ST) models by exploring model architecture design and optimization techniques tailored for LLMs. Our approach includes LLM-based speech translation architecture design, ASR-augmented training, multilingual data augmentation, and dual-LoRA optimization. Our approach demonstrates superior performance on the CoVoST-2 benchmark and showcases exceptional scaling capabilities powered by LLMs.

# Model List

| Model         | Speech Encoder       | LLM           | 
|---------------|----------------------|---------------|
| LLaST-2B      | Whisper-Large        | TinyLlama              | 
| LLaST-8B      | Whisper-Large        | Llama2-7B-Instruct     | 


# Training

## Data Preparation
- Download data from [CommonVoice](https://commonvoice.mozilla.org/en/datasets)

- Prepare tsv data as follows:

```bash
covost2/tsv
├── covost_v2.de_en.dev.tsv
├── covost_v2.de_en.test.tsv
```

- Prepare the multi-lingual data as the follows

```bash
covost/audio
├── de
├── en
├── es
├── fr
├── it
├── ja
└── zh-CN
```

- Prepare the audio data as the follows:
```bash
covost2/audio/fr/clips_16k
├── common_voice_fr_20241860.wav
├── common_voice_fr_20241864.wav
├── common_voice_fr_20241868.wav
├── common_voice_fr_20241872.wav
└── common_voice_fr_20241875.wav
```

## Installation
- It is recommeded to to build a Python-3.10 virtual environment using conda

```bash
git clone https://github.com/ParikshitGehlaut/DeepLearningProject.git
pip install -r requirements.txt
cd xtuner
pip install -e '.[all]'
```

## Training with XTuner 

```bash
export XTUNER_DATASET_TIMEOUT=120
export HF_EVALUATE_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
CUDA_VISIBLE_DEVICES=0,1 python xtuner/xtuner/tools/train.py workspace/llast_2b_tinyllama_chat.py --deepspeed deepspeed_zero2
```

## Evaluation
```bash
export HF_EVALUATE_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1 
python xtuner/tools/test.py workspace/llast_2b_tinyllama_chat.py --checkpoint work_dir/xxxx/epoch_1.pth/mp_rank_00_model_states.pt
```

# Acknowledgement

```tex
@inproceedings{chen2024llast,
  title = {LLaST: Improved End-to-end Speech Translation System Leveraged by Large Language Models},
  author = {Chen, Xi and Zhang, Songyang and Bai, Qibing and Chen, Kai and Nakamura, Satoshi},
  booktitle = {Findings of the Association for Computational Linguistics (ACL),},
  year = {2024}
}
```

# Team Members
Parikshit Gehlaut    
Soumyadeep Das    
Vemula Chandrahaas Reddy   
