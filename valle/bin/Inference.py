#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
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
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --text-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import argparse
import logging
import os
from pathlib import Path
import sys
##sys.path.append("models/VALLE/1/vallemodel/icefall")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchaudio

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import add_model_arguments, get_model
import base64

def wav_to_base64(file_path):
    with open(file_path, "rb") as wav_file:
        wav_data = wav_file.read()
        base64_data = base64.b64encode(wav_data).decode("utf-8")
        return base64_data

def base64_to_wav(base64_string, output_path):
    wav_data = base64.b64decode(base64_string.encode("utf-8"))
    with open(output_path, "wb") as wav_file:
        wav_file.write(wav_data)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-prompts",
        type=str,
        default="",
        help="Text prompts which are separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="To get up and running quickly just follow the steps below.",
        help="Text to be synthesized.",
    )

    # model
    add_model_arguments(parser)

    parser.add_argument(
        "--text-tokens",
        type=str,
        default="data/tokenized/unique_text_tokens.k2symbols",
        help="Path to the unique text tokens file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    return parser.parse_args()

'''
@torch.no_grad()
def func(input_1,input_2,input_3):
    ##args = get_args()
    text_tokenizer = TextTokenizer()
    text_tokens="data/tokenized/unique_text_tokens.k2symbols"
    text_collater = get_text_token_collater(text_tokens)
    audio_tokenizer = AudioTokenizer()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model = get_model(args)
    checkpoint=""
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model"], strict=True
        )
        assert not missing_keys
        # from icefall.checkpoint import save_checkpoint
        # save_checkpoint(f"{args.checkpoint}", model=model)

    model.to(device)
    model.eval()
    output_dir="output/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    
    text_prompts = " ".join(input_1.split("|"))

    audio_prompts = []
    
    if input_2:
        
        for n, audio_file in enumerate(input_2.split("|")):
            encoded_frames = tokenize_audio(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], 24000
                )

            audio_prompts.append(encoded_frames[0][0])

        ##assert len(args.text_prompts.split("|")) == len(audio_prompts)
        assert len(input_1.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)

    
    for n, text in enumerate(input_3.split("|")):
        logging.info(f"synthesize text: {text}")
        text_tokens, text_tokens_lens = text_collater(
            [
                tokenize_text(
                    text_tokenizer, text=f"{text_prompts} {text}".strip()
                )
            ]
        )

       
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            top_k=-100,
            temperature=1,
        )
        samples = audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])
        return samples
        
'''

def preprocess(s):
    
    s=s.strip()
    if s[-1]==".":
        s=s[:-1]
    a=s.split(".")
    for i in range(len(a)):
        a[i]=a[i].strip()
    ans=" | ".join(a)
    
    return ans

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

class Inference_Valle:
    
    def __init__(self):
        args = get_args()
        self.text_tokenizer = TextTokenizer()
        self.text_tokens="models/VALLE/1/vallemodel/valle/egs/libritts/data/tokenized/unique_text_tokens.k2symbols"
        self.text_collater = get_text_token_collater(self.text_tokens)
        self.audio_tokenizer = AudioTokenizer()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        self.model = get_model(args)
        self.checkpoint="models/VALLE/1/checkpoints/epoch-100.pt"
        if self.checkpoint:
            self.checkpoint = torch.load(self.checkpoint, map_location=self.device)
            self.missing_keys, self.unexpected_keys = self.model.load_state_dict(
                self.checkpoint["model"], strict=True
            )
            assert not self.missing_keys
        self.model.to(self.device)
        self.model.eval()
        self.output_dir="output/"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print("Done")
        
        
        
    def predict(self,input_1,input_2,input_3):
        ##formatter = (
        ##    "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        ##)
        ##logging.basicConfig(format=formatter, level=logging.INFO)
        ##valle_audio=func(input_1,input_2,input_3)
        
        ##for debug hardcode
        ##input_1="This I read with great attention, while they sat silent."
        ##input_3="This I read with great attention, while they sat silent."
        file_1 = open(input_1, "r")
        input_1=file_1.read().split("\n")[0]
        file_1.close()
        print(type(input_1))
        file_3 = open(input_3, "r")
        input_3=file_3.read().split("\n")[0]
        print(type(input_3))
        print("input_3")
        file_3.close()
        
        text_prompts = " ".join(input_1.split("|"))
        audio_prompts = []
        if input_2:
            for n, audio_file in enumerate(input_2.split("|")):
                print("audio_file---->",audio_file)
                encoded_frames = tokenize_audio(self.audio_tokenizer, audio_file)
                if False:
                    samples = audio_tokenizer.decode(encoded_frames)
                    torchaudio.save(
                        f"{args.output_dir}/p{n}.wav", samples[0], 24000
                    )
                audio_prompts.append(encoded_frames[0][0])
            assert len(input_1.split("|")) == len(audio_prompts)
            audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
            audio_prompts = audio_prompts.to(self.device)
            print(audio_prompts.shape)
            print(input_3)
            input_3=preprocess(input_3)
            for n, text in enumerate(input_3.split("|")):
                print("no of loop done ",n)
                logging.info(f"synthesize text: {text}")
                text_tokens, text_tokens_lens = self.text_collater(
                    [
                        tokenize_text(
                            self.text_tokenizer, text=f"{text_prompts} {text}".strip()
                        )
                    ]
                )

                encoded_frames = self.model.inference(
                    text_tokens.to(self.device),
                    text_tokens_lens.to(self.device),
                    audio_prompts,
                    top_k=-100,
                    temperature=1
                )
                samples = self.audio_tokenizer.decode([(encoded_frames.transpose(2, 1), None)])
                print(samples.shape)
                ##if audio_prompts != []:
                ##    print("yes")
                    ##samples = audio_tokenizer.decode(
                    ##    [(encoded_frames.transpose(2, 1), None)]
                    ##    )
                ##    print(samples.shape)
                    # store
                ##torchaudio.save("/bv4/Triton_server/audio_to_video/valle/models/VALLE/1/vallemodel/valle/egs/libritts/trial_output.wav", samples[0].cpu(), 24000)
                os.makedirs("models/VALLE/1/vallemodel/tmp_audio_folder",exist_ok=True)
                torchaudio.save("models/VALLE/1/vallemodel/tmp_audio_folder/"+str(n)+".wav", samples[0].cpu(), 24000)

                print("yes")
                
        return samples
                
            
                
        
        ###return valle_audio