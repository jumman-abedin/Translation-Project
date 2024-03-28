from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    AutoTokenizer
import numpy as np
import evaluate
from transformers import MBart50TokenizerFast
from transformers import MBartForConditionalGeneration
import sys

# select the model type from the following options (model_marianMT, model_mbart)
model_type = "marianMT"
max_input_length = 128
max_target_length = 128
source_lang = "es"
target_lang = "en"

# select the model name, name must match the model type.
model_name = "Helsinki-NLP/opus-mt-mul-en"
model_name_cleaned = model_name.replace('/', '-')

# select the dataset
dataset_name = "opus100"

# Load the data set along with soure and target language
raw_datasets = load_dataset(dataset_name, "en-es")

# Pre-process the data set
if model_type == "marianMT":
    model_marianMT = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_marianMT, use_fast=False)

elif model_type == "mbart":
    model_mbart = model_name
    tokenizer = MBart50TokenizerFast.from_pretrained(model_mbart, src_lang="en_XX", tgt_lang="de_DE")

else:
    sys.exit("Invalid Model Type, selected model from these options: {marianMT, mbart}")

# Now we will create a preprocessing function and apply it to all the data splits.
# T5 model requires a special prefix to put before the inputs, you should adopt the following code for defining the prefix. For mBART and MarianMT prefixes will remain blank.


prefix = ""  # For mBART and MarianMT prefixes will remain blank.


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# Create a subset of the data set
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Train and fine-tune the model
# We will be using AutoModelForSeq2SeqLM for T5 and MarianMT and MBartForConditionalGeneration for mBART to cache or download the models:
if model_type == "marianMT":
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

elif model_type == "mbart":
    model = MBartForConditionalGeneration.from_pretrained(model_name)

else:
    sys.exit("Invalid Model Type, selected model from these options: {marianMT, mbart}")


args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = evaluate.load("sacrebleu")
meteor = evaluate.load('meteor')


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result = {'sacrebleu': result['score']}
    result["gen_len"] = np.mean(prediction_lens)
    result["meteor"] = meteor_result["meteor"]
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

before_training = trainer.evaluate(max_length=128)

file = open(dataset_name + '-' + str(100) + '-' + model_name_cleaned +
            '. txt', 'w')

file.write(str(before_training))
file.write('\n')
file.close()

trainer.train()

after_training = trainer.evaluate(max_length=128)

file = open(dataset_name + '-' + str(100) + '-' + model_name_cleaned +
            '. txt', 'a')

file.write(str(after_training))
file.close()

trainer.save_model()
