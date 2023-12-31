
from ast import literal_eval
import logging
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import numpy as np

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.evaluate_full_model.evaluate_model import evaluate_model
from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.run_configurations import (
    RUN,
    RUN_COMMENT,
    SEED,
    PRETRAIN_WITHOUT_LM_MODEL,
    IMAGE_INPUT_SIZE,
    PERCENTAGE_OF_TRAIN_SET_TO_USE,
    PERCENTAGE_OF_VAL_SET_TO_USE,
    BATCH_SIZE,
    EFFECTIVE_BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    EVALUATE_EVERY_K_BATCHES,
    PATIENCE_LR_SCHEDULER,
    THRESHOLD_LR_SCHEDULER,
    FACTOR_LR_SCHEDULER,
    COOLDOWN_LR_SCHEDULER,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
    NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
    NUM_IMAGES_TO_PLOT,
    BERTSCORE_SIMILARITY_THRESHOLD,
    WEIGHT_OBJECT_DETECTOR_LOSS,
    WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS,
    WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS,
    WEIGHT_LANGUAGE_MODEL_LOSS,
)
from src.path_datasets_and_weights import path_full_dataset, path_runs_full_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def get_model(checkpoint=None):
    # if there is a key error when loading checkpoint, try uncommenting down below
    # since depending on the torch version, the state dicts may be different
    # checkpoint["model"]["object_detector.rpn.head.conv.weight"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.weight")
    # checkpoint["model"]["object_detector.rpn.head.conv.bias"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.bias")

    model = ReportGenerationModel(pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)
    model.to(device, non_blocking=True)

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    model.train()

    return model


def get_data_loaders(tokenizer, train_dataset, val_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    custom_collate_train = CustomCollator(tokenizer=tokenizer, is_val_or_test=False, pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)
    custom_collate_val = CustomCollator(tokenizer=tokenizer, is_val_or_test=True, pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_collate_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=custom_collate_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    # don't apply data augmentations to val set (and test set)
    val_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_transforms


def get_tokenized_datasets(tokenizer, raw_train_dataset, raw_val_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
    tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path (str)
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])
    #
    #   val dataset will have additional column:
    #   - reference_report (str)

    return tokenized_train_dataset, tokenized_val_dataset


def get_tokenizer():
    # checkpoint = "healx/gpt-2-pubmed-medium"
    # checkpoint = "/public/home/zhangke/rgrg-main/src/full_model/gpt-2-pubmed-medium"
    checkpoint = "/mnt/wsl/PHYSICALDRIVE3p1/rgrg-main/src/full_model/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_datasets(config_file_path):
    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
        "bbox_is_abnormal",
    ]

    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_labels": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval,
        "bbox_is_abnormal": literal_eval,
    }

    datasets_as_dfs = {}
    datasets_as_dfs["train"] = pd.read_csv(os.path.join(path_full_dataset, "train.csv"), usecols=usecols, converters=converters)


    usecols.append("reference_report")
    datasets_as_dfs["valid"] = pd.read_csv(os.path.join(path_full_dataset, "valid.csv"), usecols=usecols, converters=converters)

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])


    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Train: {new_num_samples_train} images")
    log.info(f"Val: {new_num_samples_val} images")

    with open(config_file_path, "a") as f:
        f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
        f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

    return raw_train_dataset, raw_val_dataset


def create_run_folder():
    """
    Run folder will contain:
        - a folder called "checkpoints" for the saved checkpoints
        - a folder called "tensorboard" for the saved tensorboard files
        - a folder called "generated_sentences_and_reports" that store the generated sentences and reports
        which were created at each evaluation
        - a txt file called "log_file", which stores information like OOMs that happened during training
        - a txt file called "run_config.txt", which stores the information specified in run_configurations.py
    """
    run_folder_path = os.path.join(path_runs_full_model, f"run_{RUN}")
    checkpoints_folder_path = os.path.join(run_folder_path, "checkpoints")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")
    generated_sentences_and_reports_folder_path = os.path.join(run_folder_path, "generated_sentences_and_reports")
    generated_sentences_folder_path = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences")
    generated_reports_folder_path = os.path.join(generated_sentences_and_reports_folder_path, "generated_reports")
    log_file = os.path.join(run_folder_path, "log_file")

    if os.path.exists(run_folder_path):
        log.error(f"Folder to save run {RUN} already exists at {run_folder_path}.")
        log.error("Delete the folder or change the run number.")
        return None

    os.mkdir(run_folder_path)
    os.mkdir(checkpoints_folder_path)
    os.mkdir(tensorboard_folder_path)
    os.mkdir(generated_sentences_and_reports_folder_path)
    os.mkdir(generated_sentences_folder_path)
    os.mkdir(generated_reports_folder_path)

    log.info(f"Run {RUN} folder created at {run_folder_path}.")

    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        "COMMENT": RUN_COMMENT,
        "SEED": SEED,
        "PRETRAIN_WITHOUT_LM_MODEL": PRETRAIN_WITHOUT_LM_MODEL,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "PERCENTAGE_OF_TRAIN_SET_TO_USE": PERCENTAGE_OF_TRAIN_SET_TO_USE,
        "PERCENTAGE_OF_VAL_SET_TO_USE": PERCENTAGE_OF_VAL_SET_TO_USE,
        "BATCH_SIZE": BATCH_SIZE,
        "EFFECTIVE_BATCH_SIZE": EFFECTIVE_BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "EVALUATE_EVERY_K_BATCHES": EVALUATE_EVERY_K_BATCHES,
        "PATIENCE_LR_SCHEDULER": PATIENCE_LR_SCHEDULER,
        "THRESHOLD_LR_SCHEDULER": THRESHOLD_LR_SCHEDULER,
        "FACTOR_LR_SCHEDULER": FACTOR_LR_SCHEDULER,
        "COOLDOWN_LR_SCHEDULER": COOLDOWN_LR_SCHEDULER,
        "NUM_BEAMS": NUM_BEAMS,
        "MAX_NUM_TOKENS_GENERATE": MAX_NUM_TOKENS_GENERATE,
        "NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE": NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
        "NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE": NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
        "NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION": NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
        "NUM_IMAGES_TO_PLOT": NUM_IMAGES_TO_PLOT,
        "BERTSCORE_SIMILARITY_THRESHOLD": BERTSCORE_SIMILARITY_THRESHOLD,
        "WEIGHT_OBJECT_DETECTOR_LOSS": WEIGHT_OBJECT_DETECTOR_LOSS,
        "WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS": WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS,
        "WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS": WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS,
        "WEIGHT_LANGUAGE_MODEL_LOSS": WEIGHT_LANGUAGE_MODEL_LOSS,
    }

    with open(config_file_path, "w") as f:
        f.write(f"RUN {RUN}:\n")
        for param_name, param_value in config_parameters.items():
            f.write(f"\t{param_name}: {param_value}\n")

    return checkpoints_folder_path, tensorboard_folder_path, config_file_path, generated_sentences_and_reports_folder_path, log_file



