from transformers import logging as hf_logging

# Modify transformers library's logging configuration
hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)
import torch
import torchaudio
from queue import Queue
from moviepy.editor import *

result_queue = Queue()

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
model = Wav2Vec2ForCTC.from_pretrained("Cattoloaffu/wav2vec2-large-xlsr-TH")

# Function to resample to 16_000
def speech_file_to_array_fn(batch, text_col="sentence", fname_col="path", resampling_to=16000):
    speech_array, sampling_rate = torchaudio.load(batch[fname_col])
    resampler = torchaudio.transforms.Resample(sampling_rate, resampling_to)
    batch["speech"] = resampler(speech_array)[0].numpy()
    batch["sampling_rate"] = resampling_to
    batch["target_text"] = batch[text_col]
    return batch

def process_sound_file(rd,data):
    try:
        wave, test_dataset = torchaudio.load(rd + "\\" + data)
        resam = torchaudio.transforms.Resample(test_dataset, 16000)
        s = resam(wave)[0].numpy()
        inputs = processor(s, sampling_rate=16_000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = ' '.join(processor.batch_decode(predicted_ids))
        
        result_queue.put((data, transcription))
    except Exception as e :
        print(f"{e}")


def generateSubtitleToFile(rd,filepath,textfilepath) :
    filenames = next(walk(rd), (None, None, []))[2]
    # Get 2 examples as sample input
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
    for data in sorted_filenames:# You can change 'train' to other splits like 'test', 'validation', etc.
        process_sound_file(rd,data)

    vc = VideoFileClip(filepath)
    shift = 4
    st = 0
    end = 5
    d = vc.audio.duration

    # Decode and print predictions
    try:
        with open(textfilepath, "a", encoding="utf-8") as f:
            while not result_queue.empty():
                data, transcription = result_queue.get()
                f.write(f"Second < {st} - {end} > :, Transcription: {transcription}\n")
                st+=shift
                end+=shift
                if end > d :
                    end = d
    except FileNotFoundError:
        print("File not Select ,and/or File Never exist")
    except AttributeError:
        print("Must import file format CORRECTLY")