# 환경 세팅
`pip install -r requirements.txt`



# 모델 추론
asr 추론: `python evaluate_salmonn.py --task asr --skip_scoring --cfg-path salmonn_eval_config_asr.yaml`

aac 추론: `python evaluate_salmonn.py --task asr --skip_scoring --cfg-path salmonn_eval_config_aac.yaml`
