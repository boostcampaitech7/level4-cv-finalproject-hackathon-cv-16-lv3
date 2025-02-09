# 안녕하세요, SOTA입니다. 👋
> **SOTA** : State-Of-The-Art 로 최신기술을 의미합니다. AI분야에서 SOTA가 되자는 의미를 담았습니다.

> 최종 발표 [구글 슬라이드]() 및 [pdf]() & [Youtube]() # 각각 어떤 것을 의도하신 건지 궁금합니다!! 유튜브를 찍어서 올리는 건가요??
> 
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">

SOTA 프로젝트는 다양한 소리를 듣고 downstream task를 수행할 수 있는 모델의 성능을 어느 정도 유지하면서 최대한 경량화 & 최적화하는 것입니다. 

베이스 모델은 [SALMONN](https://github.com/bytedance/SALMONN)입니다. 

# 모델 아키텍처
SALMONN 아키텍처
![image.png](attachment:84cb8c6b-4465-4f22-8031-d5646d8722d1:image.png)

SOTA 모델 아키텍처
![image.png](attachment:a49e33a5-e719-461e-ba8b-2fa6b014135e:image.png)

# 결과
|MODEL|SOTA|SALMONN-3B|SALMONN-7B|
|------|---|---|---|
|ASR (WER, %) ↓|테스트2|	6.34|5.1|
|AAC (SPIDEr) ↑|테스트2|27.84|48.5|
|Memory usage (MB) ↓|테스트2|9176|15750|



# 환경 세팅 및 추론
`pip install -r requirements.txt`

asr 추론: `python evaluate_salmonn.py --task asr --skip_scoring --cfg-path salmonn_eval_config_asr.yaml`

aac 추론: `python evaluate_salmonn.py --task aac --skip_scoring --cfg-path salmonn_eval_config_aac.yaml`

# demo

