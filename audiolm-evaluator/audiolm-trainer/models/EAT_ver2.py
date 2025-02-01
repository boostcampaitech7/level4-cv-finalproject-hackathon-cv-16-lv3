import torch
import torch.nn as nn
from dataclasses import dataclass
import fairseq
import torchaudio
import random
import numpy as np
import torch.nn.functional as F
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi
from typing import Optional


@dataclass
class UserDirModule:
    user_dir: str

class AudioEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        print("Initializing EAT ... ")
        model_path = UserDirModule('/root/fairseq/EAT')
        model="/root/data/_etc/_model/beats_path/EAT-base_epoch30_pt.pt"
        self.audio_width = 768 

        fairseq.utils.import_user_module(model_path)
        EATEncoder, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model])
        EATEncoder = EATEncoder[0]
        self.audio_enc = EATEncoder

                
    def forward(self, inputs, padding_mask=None):
        """
        :param inputs: audio features
        :return: encoded audio embeddings
        """
        inputs = inputs.to(dtype=torch.float16)
        audio_encoded = self.audio_enc.extract_features(inputs, padding_mask=padding_mask)['x']

        return audio_encoded
    

class EAT(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = 512
        self.encoder_embed_dim = 768
        self.post_extract_proj = ( 
            nn.Linear(self.embed, self.encoder_embed_dim)
            if self.embed != self.encoder_embed_dim
            else None
        )

        self.input_patch_size = 16
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size,
                                         bias=False)

        self.dropout_input = nn.Dropout(0.0)

        # assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = AudioEncoder() # Eat
        self.layer_norm = LayerNorm(self.embed)

        self.predictor_dropout = nn.Dropout(0.0)
        self.predictor = nn.Linear(self.encoder_embed_dim, 527)
        


    def forward_padding_mask(
            self,
            features: torch.Tensor,
            padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(
            self,
            source: torch.Tensor,
            fbank_mean: float = 15.41663,
            fbank_std: float = 6.55582,
            target_length: int = 3072,
            fixed_length = True,
            random_crop = False
    ) -> torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            # 오디오 길이를 16의 배수로 조정
            n_frames = fbank.shape[0]
        
            if not fixed_length:
                target_length = n_frames
                if target_length % 16 != 0:
                    target_length = n_frames + (16 - n_frames % 16)
            diff = target_length - n_frames
            if diff > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
                fbank = m(fbank)
            elif diff < 0:
                # 10초당 target_length가 1024이고, 모델은 30초까지 데이터를 받기 때문에 3072를 넘을 수 없음.
                print("target_length가 짱 김:", n_frames)
                # if random_crop: 
                #     start_index = random.randint(0, n_frames - target_length)
                #     fbank = fbank[:,start_index: start_index+target_length, :]
                # else: 
                #     fbank = fbank[:,0:target_length, :]

            fbanks.append(fbank)
        
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_features(
            self,
            source: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            fbank_mean: float = 15.41663,
            fbank_std: float = 6.55582,
            feature_only=False,
    ):
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std).to(torch.float32)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank) # 128 차원을 768 차원으로 확장
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)
        x = x.unsqueeze(1) # CNN에 넣기 위해 차원 추가 (B, C, H, W)의 형태. (4, 1, x, 768) 같은 형태로 변환
        x = self.encoder(
            x,
            padding_mask=padding_mask,
        )

        if not feature_only and self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)

            lprobs = torch.sigmoid(logits)

            return lprobs, padding_mask
        else:
            return x, padding_mask