# BoKA
This repo provides the code for reproducing BoKA: an automatic knowledge amalgamation framework for identifying a combination that can learn a superior student model without human labor.

### Datasets
We use two multi-domain datasets: 
  - PubMedQA (You can download it from this link: https://github.com/pubmedqa/pubmedqa)
  - MedMCQA (You can download it from this link: https://github.com/MedMCQA/MedMCQA)

### Experiments
1. To automatically select combinations, conduct the plural KA process, and evaluate the produced student model, use the following commands:
    - python bo.py (for PubMedQA)
    - python bo_medmcqa.py (for MedMCQA)
    
2. We provide the superior student models on PubMedQA and MedMCQA (https://drive.google.com/drive/folders/1UKUfop1ZXm4_V70xIv_iS6ZNpJi5Scxi?usp=sharing), respectively. The evaluation commands are as follows:
    - python evaluate.py (for both PubMedQA and MedMCQA)

