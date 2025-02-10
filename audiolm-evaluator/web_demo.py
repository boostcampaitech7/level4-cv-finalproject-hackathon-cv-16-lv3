# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from transformers import WhisperFeatureExtractor
import gradio as gr
import tempfile
import soundfile as sf
from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample
import librosa  # 16kHz 변환을 위해 추가

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--port", default=9527)
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)


# 업로드된 오디오 파일을 16kHz로 변환하는 헬퍼 함수
def convert_audio_to_16k(file_path, target_sr=16000):
    # 파일의 원본 샘플링 속도로 로드
    audio, sr = librosa.load(file_path, sr=None)
    # 이미 16kHz면 그대로, 아니면 리샘플링 수행
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

# gradio 
def gradio_reset(chat_state):
    
    chat_state = []
    return (None,
            gr.update(value=None, interactive=True),
            gr.update(placeholder='Please upload your wav first', interactive=False),
            gr.update(value="Upload & Start Chat", interactive=True),
            chat_state)

def upload_speech(gr_speech, text_input, chat_state):
    
    if gr_speech is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state.append(gr_speech)  # gr_speech는 파일 경로입니다.
    return (gr.update(interactive=False),
            gr.update(interactive=True, placeholder='Type and press Enter'),
            gr.update(value="Start Chatting", interactive=False),
            chat_state)

def gradio_ask(user_message, chatbot, chat_state):
    
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state.append(user_message)
    chatbot.append([user_message, None])
    # 현재 단일 라운드 대화만 지원합니다.
    return gr.update(interactive=False, placeholder='Currently only single round conversations are supported.'), chatbot, chat_state



# def gradio_answer(chatbot, chat_state, num_beams, temperature, top_p):
#     samples = prepare_one_sample(chat_state[0], wav_processor)
#     prompt = [
#         # cfg.config.model.prompt_template.format(chat_state[1].strip())
#         cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + chat_state[1].strip())
#     ]
#     with torch.cuda.amp.autocast(dtype=torch.float16):
#         llm_message = model.generate(
#             samples, cfg.config.generate, prompts=prompt
#         )
#     # chatbot[-1][1] = llm_message[0]
#     chatbot[-1][1] = llm_message[0].strip().removeprefix("<s>").removesuffix("</s>").strip()
#     return chatbot, chat_state

# def gradio_answer(chatbot, chat_state, num_beams, temperature, top_p):
#     # chat_state[0]에는 업로드된 오디오 파일의 경로가 들어있습니다.
#     # 업로드된 파일을 로드하고 16kHz로 변환합니다.
#     audio_16k = convert_audio_to_16k(chat_state[0], target_sr=16000)
#     # prepare_one_sample 함수는 audio array와 feature extractor를 받아 sample을 준비합니다.
#     samples = prepare_one_sample(audio_16k, wav_processor)
    
#     # prompt에 업로드된 음성에 해당하는 텍스트(채팅 입력)를 포함시킵니다.
#     prompt = [
#         cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + chat_state[1].strip())
#     ]
#     with torch.cuda.amp.autocast(dtype=torch.float16):
#         llm_message = model.generate(
#             samples, cfg.config.generate, prompts=prompt
#         )
#     # 모델 출력 후 불필요한 토큰 제거
#     chatbot[-1][1] = llm_message[0].strip().removeprefix("<s>").removesuffix("</s>").strip()
#     return chatbot, chat_state

def gradio_answer(chatbot, chat_state, num_beams, temperature, top_p):
    # chat_state[0]에는 업로드된 오디오 파일의 경로가 들어있음
    # 해당 파일을 16kHz로 변환하여 numpy array로 반환
    audio_16k = convert_audio_to_16k(chat_state[0], target_sr=16000)
    
    # 임시 파일에 16kHz 오디오 저장 (wav 형식)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_16k, 16000)
        temp_path = tmpfile.name

    # prepare_one_sample은 파일 경로를 기대하므로 임시 파일의 경로를 전달
    samples = prepare_one_sample(temp_path, wav_processor)
    
    # 채팅 입력을 prompt에 포함하여 모델 추론
    prompt = [
        cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + chat_state[1].strip())
    ]
    with torch.amp.autocast("cuda",dtype=torch.float16):
        llm_message = model.generate(
            samples, cfg.config.generate, prompts=prompt
        )
    chatbot[-1][1] = llm_message[0].strip().removeprefix("<s>").removesuffix("</s>").strip()
    return chatbot, chat_state

title = """<h1 align="center"> 🚀SOTA 팀입니다.</h1>"""
image_src = """<h1 align="center"><a href="https://github.com/boostcampaitech7/level4-cv-finalproject-hackathon-cv-16-lv3"><img src="data:image/jpeg;base64,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", alt="SALMONN" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>"""
description = """<h3>AI boostcamp SOTA팀의 데모입니다. 여러분의 오디오파일(.wav)를 업로드하고 AI와 채팅해보세요!</h3>"""


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(image_src)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            speech = gr.Audio(label="Audio", type='filepath')
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                interactive=True,
                label="beam search numbers",
            )

            # top_p = gr.Slider(
            #     minimum=0.1,
            #     maximum=1.0,
            #     value=0.9,
            #     step=0.1,
            #     interactive=True,
            #     label="top p",
            # )

            # temperature = gr.Slider(
            #     minimum=0.8,
            #     maximum=2.0,
            #     value=1.0,
            #     step=0.1,
            #     interactive=False,
            #     label="temperature",
            # )

        with gr.Column():
            chat_state = gr.State([])
            
            chatbot = gr.Chatbot(label='SALMONN')
            text_input = gr.Textbox(label='User', placeholder='Please upload your speech first', interactive=False)

    with gr.Row():
        examples = gr.Examples(
            examples = [
                ["resource/audio_demo/gunshots.wav", "Recognize the speech and give me the transcription."],
                ["resource/audio_demo/gunshots.wav", "Provide the phonetic transcription for the speech."],
                ["resource/audio_demo/gunshots.wav", "Please describe the audio."],
                ["resource/audio_demo/gunshots.wav", "Recognize what the speaker says and describe the background audio at the same time."],
                ["resource/audio_demo/gunshots.wav", "Please answer the speaker's question in detail based on the background sound."],
                ["resource/audio_demo/duck.wav", "Please list each event in the audio in order."],
                ["resource/audio_demo/duck.wav", "Based on the audio, write a story in detail. Your story should be highly related to the audio."],
                ["resource/audio_demo/duck.wav", "How many speakers did you hear in this audio? Who are they?"],
                ["resource/audio_demo/excitement.wav", "Describe the emotion of the speaker."],
                ["resource/audio_demo/mountain.wav", "Please answer the question in detail."],
                ["resource/audio_demo/music.wav", "Please describe the music in detail."],
                ["resource/audio_demo/music.wav", "What is the emotion of the music? Explain the reason in detail."],
                ["resource/audio_demo/music.wav", "Can you write some lyrics of the song?"],
                ["resource/audio_demo/music.wav", "Give me a title of the music based on its rhythm and emotion."]
            ],
            inputs=[speech, text_input]
        )
        
    upload_button.click(upload_speech, [speech, text_input, chat_state], [speech, text_input, upload_button, chat_state])

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state], [chatbot, chat_state]
    )
    clear.click(gradio_reset, [chat_state], [chatbot, speech, text_input, upload_button, chat_state], queue=False)



# demo.launch(share=True, enable_queue=True, server_port=int(args.port))
demo.launch(share=True, server_port=int(args.port))
