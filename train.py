debug = True
SEED = 42


from typing import Optional, Tuple, Union, List
import numpy as np
from math import inf as INF
import argparse
import os, sys
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from datasets import Dataset, load_dataset
import evaluate

sys.path.append("../")
from utils import get_tokenizer
# from utils.variables import LANGS
# LANGS = sorted(LANGS)
# print(LANGS)

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations import TensorBoardCallback, is_tensorboard_available
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


from datasets import load_dataset, interleave_datasets, concatenate_datasets
from evaluate import load


def init_tokenizer(TOKENIZER_INPATH, FILES):
    '''Note that if we are using a pretrained tokenizer,
    we should simply pass the HF key of the model as TOKENIZER_INPATH'''

    # If we are using a pretrained tokenizer, this is the same as (e.g.):
    # tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")

    logging.info("Loading src tokenizer from {}".format(TOKENIZER_INPATH))
    tokenizer = get_tokenizer.train_or_load_tokenizer(TOKENIZER_INPATH,  \
        FILES = FILES)
    # Looks like HF MT doesn't support separate source and target tokenizers
    # logging.info("Loading tgt tokenizer from {}".format(TGT_TOKENIZER_INPATH))
    # tgt_tokenizer = get_tokenizer.train_or_load_tokenizer(TGT_TOKENIZER_INPATH, tokenizer)

    ### Optionally add language ID tokens
    # tokenizer = get_tokenizer.add_langid_tokens(tokenizer, LANGS)

    
    return tokenizer

def init_models(MODELPATH, tokenizer, PT_CKPT = None):
    '''Get LM model'''

    # Initialize Seq2Seq model, input and output tokenizer, special hyperparameters
    if PT_CKPT or MODELPATH:
        # First we check if there is some checkpoint OR HF model key 
        logging.info("Loading model from {}".format(PT_CKPT))
        model = AutoModelForCausalLM.from_pretrained(PT_CKPT)
    else:
        # If not, we initialize the encoder and decoder from scratch
        logging.info("Initializing encoder-decoder model from scratch")
        config = GPT2Config(vocab_size=len(tokenizer), n_layer=6, n_head=4, n_embd=512)
        model = GPT2LMHeadModel(config)
        

    ## Set model parameters
    # model_enc_dec.model_lid = model_lid
    # model_enc_dec.alpha = alpha
    # model_enc_dec.tau = tau
    # model_enc_dec.istauhard = istauhard
    if model.config.vocab_size !=len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    return model

def get_multilingual_dataset_from_files(DATAFILES, max_lines, tokenizer, max_length = 512):
    '''
    The function assumes that DATAFILES is a list of paths to files, where each file is a dataset
    in a different language. We will perform the train test split for each language, and then
    interleave the datasets with upsampling based on the proportion of examples in each dataset.
    Args
        - DATAFILES: list of paths to files
        - max_lines: maximum number of lines to load
        - tokenizer: tokenizer to use
        - max_length: maximum length of sequence
    Returns
        - train_dataset, dev_dataset, test_dataset
    '''

    all_train_datasets = list()
    all_dev_datasets = list()
    all_test_datasets = list()
    for datafile in DATAFILES:
        logging.info("Loading : {} ".format(datafile))
        data = load_dataset("text", data_files={"train": [datafile]})
        data = data.select(range(max_lines))
        dataset = dataset.train_test_split(test_size=0.1, shuffle = True, seed=SEED)
        train_dataset = dataset["train"]
        devtest_dataset = dataset["test"]
        devtest_dataset = devtest_dataset.train_test_split(test_size=0.5,shuffle = True, seed=SEED)
        dev_dataset = devtest_dataset["train"]
        test_dataset = devtest_dataset["test"]
        all_train_datasets.append(train_dataset)
        all_dev_datasets.append(dev_dataset)
        all_test_datasets.append(test_dataset)


    def find_probs(S = 0.7):
        all_lengths = [len(dataset) for dataset in all_train_datasets]
        logging.info("LEN OF DATASETS: {}".format(str(all_lengths)))
        norm_lengths = [length/sum(all_lengths) for length in all_lengths]
        probs = [elem**S for elem in norm_lengths]
        norm_probs = [prob/sum(probs) for prob in probs]
        logging.info("PROBS: {}".format(str(norm_probs)))
        return norm_probs

    if len(all_train_datasets) > 1:
        probs = find_probs()

        train_dataset = interleave_datasets(all_train_datasets, probabilities=probs, seed=SEED, \
            stopping_strategy="all_exhausted")
        dev_dataset = concatenate_datasets(all_dev_datasets)
        test_dataset = concatenate_datasets(all_test_datasets)
    else:
        train_dataset = all_train_datasets[0]
        dev_dataset = all_dev_datasets[0]
        test_dataset = all_test_datasets[0]

    # Log all sizes
    # logging.info("Length of dataset: {}".format(len(dataset)))
    logging.info("Length of train dataset: {}".format(len(train_dataset)))
    logging.info("Length of dev dataset: {}".format(len(dev_dataset)))
    logging.info("Length of test dataset: {}".format(len(test_dataset)))

    # Preprocess
    tokenize = lambda x: tokenizer(x["text"], truncation=True, max_length=max_length)
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    dev_dataset = dev_dataset.map(tokenize, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["text"])

    logging.info("DONE TOKENIZING!")

    train_data = train_data.with_format("torch")
    dev_data = dev_data.with_format("torch")
    test_data = test_data.with_format("torch")
    
    return train_dataset, dev_dataset, test_dataset


def get_multilingual_dataset(DATADIRS, max_lines, tokenizer, max_length = 512):

    '''
    This function assumes that each datadir in DATADIRS contains train, dev, and test splits, 
    called "train", "dev", and "test".
    We will load each of these datasets, and then interleave them with upsampling based on the
    proportion of examples in each dataset.
    Note: This also works with just one dataset.
    Note2: We use the same tokenizer for all languages.

    Args
        - DATADIRS: list of paths to directories containing train, dev, and test splits
        - max_lines: maximum number of lines to load
        - tokenizer: tokenizer to use
        - max_length: maximum length of sequence
    Returns
        - train_dataset, dev_dataset, test_dataset
    '''
    ## Split dataset into train, dev, and test if no splits are provided
    # dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
    # train_dataset = dataset["train"]
    # devtest_dataset = dataset["test"]
    # devtest_dataset = devtest_dataset.train_test_split(test_size=0.5, seed=SEED)
    # dev_dataset = devtest_dataset["train"]
    # test_dataset = devtest_dataset["test"]
    # splits = {"train", "test", "dev"}

    TRAIN_FILES = [os.path.join(datadir, "train") for datadir in DATADIRS]
    DEV_FILES = [os.path.join(datadir, "dev") for datadir in DATADIRS]
    TEST_FILES = [os.path.join(datadir, "test") for datadir in DATADIRS]

    all_datasets = list()

    for idx, _ in enumerate(TRAIN_FILES):
        logging.info("Loading : {} ".format(TRAIN_FILES[idx]))
        data = load_dataset("text", data_files={"train": [TRAIN_FILES[idx]], \
        "dev": [DEV_FILES[idx]], "test": [TEST_FILES[idx]]})
        data = data.select(range(max_lines))
        all_datasets.append(data)

    
    all_train_datasets = [dataset["train"] for dataset in all_datasets]
    all_dev_datasets = [dataset["dev"] for dataset in all_datasets]
    all_test_datasets = [dataset["test"] for dataset in all_datasets]

    def find_probs(S = 0.7):
        all_lengths = [len(dataset) for dataset in all_train_datasets]
        logging.info("LEN OF DATASETS: {}".format(str(all_lengths)))
        norm_lengths = [length/sum(all_lengths) for length in all_lengths]
        probs = [elem**S for elem in norm_lengths]
        norm_probs = [prob/sum(probs) for prob in probs]
        logging.info("PROBS: {}".format(str(norm_probs)))
        return norm_probs

    if len(all_train_datasets) > 1:
        probs = find_probs()

        train_dataset = interleave_datasets(all_train_datasets, probabilities=probs, seed=SEED, \
            stopping_strategy="all_exhausted")
        dev_dataset = concatenate_datasets(all_dev_datasets)
        test_dataset = concatenate_datasets(all_test_datasets)
    else:
        train_dataset = all_train_datasets[0]
        dev_dataset = all_dev_datasets[0]
        test_dataset = all_test_datasets[0]

    # Log all sizes
    # logging.info("Length of dataset: {}".format(len(dataset)))
    logging.info("Length of train dataset: {}".format(len(train_dataset)))
    logging.info("Length of dev dataset: {}".format(len(dev_dataset)))
    logging.info("Length of test dataset: {}".format(len(test_dataset)))

    # Preprocess
    tokenize = lambda x: tokenizer(x["text"], truncation=True, max_length=max_length)
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    dev_dataset = dev_dataset.map(tokenize, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["text"])

    logging.info("DONE TOKENIZING!")

    train_data = train_data.with_format("torch")
    dev_data = dev_data.with_format("torch")
    test_data = test_data.with_format("torch")
    
    return train_dataset, dev_dataset, test_dataset

def compute_metrics(pred):
    '''Compute perplexity'''
    global perplexity
    preds = pred.predictions

    results = perplexity.compute(predictions=preds, model_id="gpt2")
    return results


class CustomTensorboardCallback(TensorBoardCallback):
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''We visualize things here'''
        model = kwargs.pop("model")
        something_to_visualize = model.log["something_to_visualize"]

        print(f"In custom callback, something to visualize: {something_to_visualize}")

        # Log to tensorboard
        self.tb_writer.add_scalar("something_to_visualize", something_to_visualize, state.global_step)


def main(args):

    global tokenizer
    global perplexity

    # if "spanish" in args.ENC_DEC_MODELPATH:
    #     script = "lat"
    # elif "hindi" in args.ENC_DEC_MODELPATH:
    #     script = "dev"

    # Get seq2seq model and tokenizer
    logging.info("Initializing tokenizer...")
    FILES = [os.path.join(datadir, "train") for datadir in args.DATADIRS]
    tokenizer = init_tokenizer(args.TOKENIZER_INPATH, \
                               FILES)

    logging.info("Initializing models...")
    model = init_models(args.MODELPATH, tokenizer, args.PT_CKPT)
    
    logging.info("Getting datasets...")
    # Get dataset splits, and preprocess them
    ## If DATADIRS is a list of dirs, we've already split the data
    if os.isdir(args.DATADIRS[0]):
        train_dataset, dev_dataset, test_dataset = \
        get_multilingual_dataset(args.DATADIRS, max_lines=args.max_lines, tokenizer= tokenizer, max_length= args.max_length)
    else:
    ## If DATADIRS is a list of files, we need to split the data
        train_dataset, dev_dataset, test_dataset = \
        get_multilingual_dataset_from_files(args.DATADIRS, max_lines=args.max_lines, tokenizer= tokenizer, max_length= args.max_length)
    
    # Instead of that, download some MT dataset from HF
    # tokenizer = AutoTokenizer.from_pretrained(args.ENC_DEC_MODELPATH)
    # train_dataset = load_dataset("wmt16", "de-en", split="train[:1%]").select(range(args.max_lines))
    # dev_dataset = load_dataset("wmt16", "de-en", split="test[:1%]").select(range(int(args.max_lines/9))) # This gives us a 9:1 ratio
    # train_dataset, dev_dataset = preprocess_wmt(train_dataset, dev_dataset, tokenizer, max_length = args.max_length)
    # test_dataset = dev_dataset
    # Print example
    logging.info(f"EXAMPLE: {train_dataset[0]} ")   

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Metric
    perplexity = evaluate.load("perplexity")

    # Initialize Seq2SeqTrainer
    logging.info("Initializing trainer...")

    # if args.resume_from_checkpoint: train_steps = 1
    # else: train_steps = len(train_dataset) * args.epochs // args.batch_size

    training_args = TrainingArguments(
    output_dir=args.OUTPUT_DIR,
    resume_from_checkpoint=args.resume_from_checkpoint,
    overwrite_output_dir=False,
    num_train_epochs=args.epochs,
    # max_steps=train_steps,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=args.LOG_DIR,
    predict_with_generate=True,
    generation_max_length=40, # defaults to model config max_length
    report_to="tensorboard",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    # eval_steps=2000, 
    # logging_steps=2000,
    # save_steps=2000, # For 15000 examples, this will save roughly every epoch with batch size 8
    load_best_model_at_end=True,
    save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator= data_collator,
        compute_metrics=compute_metrics,
        callbacks=[CustomTensorboardCallback],
    )   

    
    logging.info("STARTING TRAINING")
    logging.info(f"CUDA: {torch.cuda.is_available()}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logging.info("SAVING MODEL")
    model.save_pretrained(args.OUTPUT_DIR)

    # # Get performance and labels on test set
    if test_dataset:
        logging.info("STARTING EVALUATION")
        test_results = trainer.predict(test_dataset)
        test_metrics = test_results.metrics
        predictions = test_results.predictions
        labels = test_results.label_ids 
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode into text
        inputs = tokenizer.batch_decode(test_dataset["input_ids"], skip_special_tokens=True)
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Log examples
        logging.info("Logging examples...")
        for i in range(len(predictions[:10])):
            logging.info("Example {}: ".format(i))
            logging.info("Input: {}".format(inputs[i]))
            logging.info("Prediction: {}".format(predictions[i]))
            logging.info("Label: {}".format(labels[i]))
        # Log metrics
        logging.info("Logging metrics...")
        logging.info("Test metrics: {}".format(test_metrics))
        logging.info("DONE EVALUATION")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATADIRS", type=str, default=None)
    parser.add_argument("--MODELPATH", type=str, default=None, help="Path to HF model (e.g. ai-forever/mGPT)")
    parser.add_argument("--TOKENIZER_INPATH", type=str, default=None, help="Path to tokenizer - if self-trained, put path. If None, \
                        the tokenizer from the encoder model will be used")
    parser.add_argument("--PT_CKPT", type=str, default=None, help="Path to PGN checkpoint")
    parser.add_argument("--max_length", type=int, default = 512)
    parser.add_argument("--OUTPUT_DIR", type=str, default="output_dir", help="Path to save model")
    parser.add_argument("--LOG_DIR", type=str, default="logs", help="Path to save tensorboard logs")
    parser.add_argument("--epochs", type=int, default = 20)
    parser.add_argument("--batch_size", type=int, default = 16)
    parser.add_argument("--max_lines", type=int, default = INF)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False, help="Resume training from args.OUTPUT_DIR")
    # Take any additional approach-related parameters

    args = parser.parse_args()

    logging.basicConfig(filename=f"{args.LOG_DIR}/log.txt", filemode="w", format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)


    main(args)