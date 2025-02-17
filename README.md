# 안녕하세요, SOTA입니다. 👋
> **SOTA** : State-Of-The-Art 로 최신기술을 의미합니다. AI분야에서 SOTA가 되자는 의미를 담았습니다.

> 최종 발표 [Report](src/Nota_CV_팀리포트(16조).pdf) 및 [발표 자료](src/CV_16조_NOTA_발표자료.pdf)
> 
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">

SOTA 프로젝트는 다양한 소리를 듣고 downstream task를 수행할 수 있는 모델의 성능을 어느 정도 유지하면서 최대한 경량화 & 최적화하는 것입니다. 

베이스 모델은 [SALMONN](https://github.com/bytedance/SALMONN)입니다. 

## 팀원 소개
|[김동욱](https://github.com/82153)|[김재진](https://github.com/jeajin)|[이재건](https://github.com/2JAE22)|[박정욱](https://github.com/wjddnr0920)|[황은섭](https://github.com/eunseopski)|
|---|---|---|---|---|
|<div align="center">![Image](https://github.com/user-attachments/assets/7962a4ef-1901-4603-9b73-331a0a8f0a10)</div>|<div align="center">![Image](https://github.com/user-attachments/assets/31b5e793-6407-4a79-a2a1-fe2df3f70b5a)</div>|<div align="center">![Image](https://github.com/user-attachments/assets/bb43a22f-650b-48a6-ac05-b2caa57d4686)</div>|<div align="center">![Image](https://github.com/user-attachments/assets/28eeeae0-a54b-4818-8e24-1b89ecac0cb6)</div>|<div align="center">![Image](https://github.com/user-attachments/assets/06722555-5806-47fe-a2e6-ee33da5e5375)</div>|
|EDA|개발 환경 구축 및 초기 세팅|개발 환경 구축 및 초기 세팅|EDA|모델 학습 및 실험 관리|
|경량화 기법 및 모델 서칭|BaseLine 코드 및 모델 분석|모델 학습 및 실험 관리|오디오 증강 리서치|BaseLine 코드 및 모델 분석|
|BaseLine 코드 및 모델 분석|Dataset 구축|경량화 기법 및 모델 서칭|BaseLine 코드 및 모델 분석|Dataset 구축|

# demo
![Demo](src/demo.gif)

# 모델 아키텍처
SALMONN 아키텍처
<img src="src/SALMONN.png">

SOTA 모델 아키텍처
<img src="src/SOTA.png">


# 결과
|MODEL|SOTA|SALMONN-3B|SALMONN-7B|
|------|---|---|---|
|ASR (WER, %) ↓|5.3|	6.34|5.1|
|AAC (SPIDEr) ↑|35.49|27.84|48.5|
|Memory usage (MB) ↓|4095|9176|15750|




# 환경 세팅 및 추론
`pip install -r requirements.txt`

asr 추론: `python evaluate_salmonn.py --task asr --skip_scoring --cfg-path salmonn_eval_config_asr.yaml`

aac 추론: `python evaluate_salmonn.py --task aac --skip_scoring --cfg-path salmonn_eval_config_aac.yaml`



