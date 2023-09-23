from functools import partial
import pandas as pd
import numpy as np
from datasets import (
    load_dataset, 
    load_from_disk,
    load_metric,)
# from datasets.filesystems import S3FileSystem
from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import torchaudio
import re
import json
from pythainlp.tokenize import word_tokenize, thai_syllables
import random
from IPython.display import display, HTML
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


torch.cuda.empty_cache()
cv_root: str = "D:\\UT\\DataSet"


def preprocess_data(example, tok_func = word_tokenize):
    example['sentence'] = ' '.join(tok_func(example['sentence']))
    return example

def speech_file_to_array_fn(batch, 
                            text_col="sentence", 
                            fname_col="path",
                            resampling_to=16000):
    speech_array, sampling_rate = torchaudio.load(batch[fname_col])
    resampler=torchaudio.transforms.Resample(sampling_rate, resampling_to)
    batch["speech"] = resampler(speech_array)[0].numpy()
    batch["sampling_rate"] = resampling_to
    batch["target_text"] = batch[text_col]
    return batch

def extract_all_chars(batch, text_col = "sentence"):
    all_text = " ".join(batch[text_col])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(df)


print("Loading Data...")
datasets = load_dataset("D:\\UT\\Scripts\\th_common_voice_70.py", "th")
datasets = datasets.map(preprocess_data)

vocabs = datasets.map(extract_all_chars,
                      batched=True,
                      batch_size=-1, 
                      keep_in_memory=True, 
                      remove_columns=datasets.column_names["train"])

vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["validation"]["vocab"][0]) |set(vocabs["test"]["vocab"][0]))
# vocab_list = list(set(vocabs["train"]["vocab"][0])) #strictly no leakage
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

print (len(vocab_dict))
print(vocab_dict)

#make space = |
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]


#padding token serves as blank token
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict), vocab_dict


print("Making JSON...")
with open(f"{cv_root}\\vocab.json", 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer(f"{cv_root}\\vocab.json", 
                                 unk_token="[UNK]", 
                                 pad_token="[PAD]", 
                                 word_delimiter_token="|")
tokenizer.save_pretrained('D:\\UT\\pretrained')