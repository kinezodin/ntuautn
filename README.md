# ntuautn: IEEE COVID-19 Public Health Informatics Challenge
---

## Abstract

The COVID-19 pandemic has brought forth many challenges to the scientific community, from testing, diagnosing and contact tracing, to prevention, treatment and identification of risk factors. Over the past decade there have been significant technological advancements, especially in the area of Artificial Intelligence and Deep Learning, which motivates researchers to find novel solutions to the aformentioned challenges by making use of these technologies. In this context, the IEEE COVID-19 Sensor Informatics Challenge poses the question of whether audio data can be utilized for diagnosing the disease, without sacrificing interpretability. In this work we propose a partially interpretable pipeline based on convolutional neural networks for determining if a subject is likely to have COVID, based on audio samples of their breathing, speech and cough, which were provided in the context of the challenge. 


## Usage 
---
### Environment Setup

1. Make sure you have Anaconda or Miniconda installed
2. Clone repo with git clone https://github.com/kinezodin/ntuautn.git
3. Go into the cloned repo: cd ntuautn
4. Create the environment: conda env create -f environment.yml
5. Activate the environment: conda activate ntuatn

### Pipeline
---
- First download the competition dataset
- Then run prepare_features.py to create mel-spectrograms from the data
- Run train.py to train on one of the 5 folds (specified through the variable 'fold')
- Or download the pre-trained (on fold 4) weights and run evaluate.py or interpret.py


### Download Pretrained Weights

Link for the pretrained weights: https://drive.google.com/u/7/uc?export=download&confirm=An4_&id=1lTdSJdpn6qtGpc9obudfCnxiPfODJB_d



