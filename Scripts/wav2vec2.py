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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

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

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    max_length = 200
    
    batch["input_values"] = processor(batch["speech"], 
                                      sampling_rate=batch["sampling_rate"][0],
                                      padding=True,
                                      ).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"],padding=True,truncation=True,).input_ids
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
    
# show_random_elements(datasets["train"].remove_columns(["path"]), num_examples=20)

def compute_metrics(pred, processor, metric):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


# @dataclass
# class DataCollatorCTCWithPadding:
#     """
#     Data collator that will dynamically pad the inputs received.
#     Args:
#         processor (:class:`~transformers.Wav2Vec2Processor`)
#             The processor used for proccessing the data.
#         padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
#             Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
#             among:
#             * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
#               sequence if provided).
#             * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
#               maximum acceptable input length for the model if that argument is not provided.
#             * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
#               different lengths).
#         max_length (:obj:`int`, `optional`):
#             Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
#         max_length_labels (:obj:`int`, `optional`):
#             Maximum length of the ``labels`` returned list and optionally padding length (see above).
#         pad_to_multiple_of (:obj:`int`, `optional`):
#             If set will pad the sequence to a multiple of the provided value.
#             This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
#             7.5 (Volta).
#     """

#     processor: Wav2Vec2Processor
#     padding: Union[bool, str] = True
#     max_length: Optional[int] = None
#     max_length_labels: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     pad_to_multiple_of_labels: Optional[int] = None

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lenghts and need
#         # different padding methods
#         input_features = [{"input_values": feature["input_values"]} for feature in features]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]

#         batch = self.processor.pad(
#             input_features,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )
#         with self.processor.as_target_processor():
#             labels_batch = self.processor.pad(
#                 label_features,
#                 padding=self.padding,
#                 max_length=self.max_length_labels,
#                 pad_to_multiple_of=self.pad_to_multiple_of_labels,
#                 return_tensors="pt",
#             )

#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

#         batch["labels"] = labels

#         return batch


print("Loading Data...")
datasets = load_dataset("D:\\UT\\Scripts\\th_common_voice_70.py", "th",cache_dir="D:\\UT\\cache")
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

print("Making JSON...")
# with open(f"{cv_root}\\vocab.json", 'w') as vocab_file:
#     json.dump(vocab_dict, vocab_file)

# tokenizer = Wav2Vec2CTCTokenizer(f"{cv_root}\\vocab.json", 
#                                  unk_token="[UNK]", 
#                                  pad_token="[PAD]", 
#                                  word_delimiter_token="|")
# tokenizer.save_pretrained('D:\\UT\\pretrained')

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")

featureExtractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                            sampling_rate=16000,
                                            padding_value=0.0,
                                            do_normalize=True,
                                            return_attention_mask=False,
                                            cache_dir="D:\\UT\\cache"
                                            )


processor = Wav2Vec2Processor(feature_extractor=featureExtractor,tokenizer=tokenizer)

print("Prepare speech_dataset...")
speech_datasets = datasets.map(speech_file_to_array_fn, 
                                remove_columns=datasets.column_names["train"])
print("Prepare completed...")


prepared_datasets = speech_datasets.map(prepare_dataset, 
                                        remove_columns=speech_datasets.column_names["train"], 
                                        batch_size=16,
                                        batched=True,
                                        )
print(prepared_datasets)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = load_metric("wer")

print(data_collator)


# Create Model for training
model = Wav2Vec2ForCTC.from_pretrained("E:\\Model\\checkpoint-468000")


model.freeze_feature_extractor()
training_args = TrainingArguments(
    output_dir="E:\\Model",
    group_by_length=True,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    metric_for_best_model='wer',
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    num_train_epochs=1,
    fp16=True,
    learning_rate=1e-4,
    warmup_steps=1000,
    save_total_limit=1,
    report_to="tensorboard",
)

print('Training model....')
# Train
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=partial(compute_metrics, metric=wer_metric, processor=processor),
    train_dataset=prepared_datasets["train"],
    eval_dataset=prepared_datasets["validation"],
    tokenizer=processor.feature_extractor,
    )

torch.cuda.empty_cache()

trainer.train()
print("Training completed...")