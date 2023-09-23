from datasets import load_dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
import torch
import torchaudio

from moviepy.editor import *

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)
from os import walk

# Load pretrained processor and model
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = Wav2Vec2ForCTC.from_pretrained("D:\\UT\\Model2\\checkpoint-468000")

# Function to resample to 16_000
def speech_file_to_array_fn(batch, text_col="sentence", fname_col="path", resampling_to=16000):
    speech_array, sampling_rate = torchaudio.load(batch[fname_col])
    resampler = torchaudio.transforms.Resample(sampling_rate, resampling_to)
    batch["speech"] = resampler(speech_array)[0].numpy()
    batch["sampling_rate"] = resampling_to
    batch["target_text"] = batch[text_col]
    return batch

# Load multiple datasets
# Replace "your_dataset_name" with the actual name of the dataset
def generateSubtitleToFile(rd,filepath,textfilepath) :
    filenames = next(walk(rd), (None, None, []))[2]
    # Get 2 examples as sample input
    inputs_list = []
    for data in filenames:# You can change 'train' to other splits like 'test', 'validation', etc.
        wave,test_dataset = torchaudio.load(rd+"\\"+data)
        resam = torchaudio.transforms.Resample(test_dataset,16000)
        s = resam(wave)[0].numpy()
        inputs = processor(s, sampling_rate=16_000, return_tensors="pt", padding=True)
        inputs_list.append(inputs)
        
    # Infer
    with torch.no_grad():
        logits_list = [model(inputs.input_values).logits for inputs in inputs_list]

    predicted_ids_list = [torch.argmax(logits, dim=-1) for logits in logits_list]

    vc = VideoFileClip(filepath)
    shift = 4
    st = 0
    end = 5
    d = vc.audio.duration

    # Decode and print predictions
    for predicted_ids in predicted_ids_list:
        f = open(textfilepath,"a",encoding="utf-8")
        s = ' '.join(processor.batch_decode(predicted_ids))
        f.write("Second <"+str(st) +"- " +str(end) +"> : "+s+"\n")
        f.close()
        st+=shift
        end+=shift
        if end > d :
            end = d