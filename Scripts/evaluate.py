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

cv_root: str = "D:\\UT\\DataSet"

def evaluate(batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values.to(device),).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_sentence"] = processor.batch_decode(pred_ids)
    return batch

wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

datasets = load_dataset("D:\\UT\\Scripts\\th_common_voice_70.py", "th",split="test")
print(datasets)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, 
                                             sampling_rate=16000, 
                                             padding_value=0.0, 
                                             do_normalize=True, 
                                             return_attention_mask=False)


processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = Wav2Vec2ForCTC.from_pretrained("E:\\Model\\checkpoint-468000")

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

test_dataset = datasets.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values,).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])

# result = test_dataset.map(evaluate, batched=True, batch_size=8)
# result_df = pd.DataFrame({'sentence':result['sentence'].replace(' ',''), 
#                            'pred_sentence_tok': result['pred_sentence'].replace(' ','')})
# result_df['sentence_tok'] = result_df.sentence.map(lambda x: ' '.join(word_tokenize(x)))
# result_df['pred_sentence'] = result_df.pred_sentence_tok.map(lambda x: ''.join(x.split()))
# #change tokenization to fit pythainlp tokenization
# result_df['pred_sentence_tok'] = result_df.pred_sentence.map(lambda x: ' '.join(word_tokenize(x)))

# from pythainlp.spell import spell_sent

# result_df['pred_sentence_tok_corrected'] = result_df['pred_sentence_tok']\
#     .map(lambda x: ' '.join(spell_sent(x.split(), engine='sympellpy')[0]))
# result_df['pred_sentence_corrected'] = result_df['pred_sentence_tok_corrected']\
#     .map(lambda x: ''.join(x.split()))
# result_df['pred_sentence_tok_corrected'] = result_df.pred_sentence_corrected.map(lambda x: ' '.join(word_tokenize(x)))

# result_df.to_csv('artifacts/result_cv70.csv',index=False)

# wer_metric.compute(predictions=result_df.pred_sentence_tok,references=result_df.sentence_tok)