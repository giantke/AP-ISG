from collections import defaultdict
import csv
import io
import os
import re
import tempfile

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import spacy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from tqdm import tqdm

from src.CheXbert.src.constants import CONDITIONS
from src.CheXbert.src.label import label
from src.CheXbert.src.models.bert_labeler import bert_labeler
from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.evaluate_full_model.cider.cider import Cider
from src.full_model.run_configurations import (
    BATCH_SIZE,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
    NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
    NUM_IMAGES_TO_PLOT,
    BERTSCORE_SIMILARITY_THRESHOLD,
)
from src.path_datasets_and_weights import path_chexbert_weights


import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bert_score_source.bert_score import score as bert_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_NLG_scores(nlg_metrics: list[str], gen_sents_or_reports: list[str], ref_sents_or_reports: list[str]) -> dict[str, float]:
    def convert_for_pycoco_scorer(sents_or_reports: list[str]):
        sents_or_reports_converted = {}
        for num, text in enumerate(sents_or_reports):
            sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " ."))]

        return sents_or_reports_converted
    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    if "meteor" in nlg_metrics:
        scorers["meteor"] = Meteor()
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()  # this is actually the Rouge-L score, even if the class name only says Rouge
    if "cider" in nlg_metrics:
        scorers["cider"] = Cider()  # this is actually the Cider-D score, even if the class name only says Cider

    gen_sents_or_reports = convert_for_pycoco_scorer(gen_sents_or_reports)
    ref_sents_or_reports = convert_for_pycoco_scorer(ref_sents_or_reports)

    nlg_scores = {}

    for metric_name, scorer in scorers.items():
        score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score

    return nlg_scores


def compute_clinical_efficacy_scores(language_model_scores: dict, gen_reports: list[str], ref_reports: list[str]):
    def get_chexbert():
        # Specify the path of pretrained labeler
        model = bert_labeler()
        model = nn.DataParallel(model)  # needed since weights were saved with nn.DataParallel
        checkpoint = torch.load(path_chexbert_weights, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        return model

    def get_chexbert_labels_for_gen_and_ref_reports():
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")
            csv_ref_reports_file_path = os.path.join(temp_dir, "ref_reports.csv")

            header = ["Report Impression"]

            with open(csv_gen_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[gen_report] for gen_report in gen_reports])

            with open(csv_ref_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[ref_report] for ref_report in ref_reports])

            # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
            preds_gen_reports = label(chexbert, csv_gen_reports_file_path)
            preds_ref_reports = label(chexbert, csv_ref_reports_file_path)

        return preds_gen_reports, preds_ref_reports

    def compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports):
        def convert_labels_like_miura(preds_reports: list[list[int]]):
            def convert_label(label: int):
                if label == 2:
                    return 0
                elif label == 3:
                    return 1
                else:
                    return label

            preds_reports_converted = [[convert_label(label) for label in condition_list] for condition_list in preds_reports]

            return preds_reports_converted

        preds_gen_reports_converted = convert_labels_like_miura(preds_gen_reports)
        preds_ref_reports_converted = convert_labels_like_miura(preds_ref_reports)

        five_conditions_to_evaluate = {"Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"}

        total_preds_gen_reports_5_conditions = []
        total_preds_ref_reports_5_conditions = []

        # we also compute the micro average over all 14 conditions:
        total_preds_gen_reports_14_conditions = []
        total_preds_ref_reports_14_conditions = []

        # iterate over the 14 conditions
        for preds_gen_reports_condition, preds_ref_reports_condition, condition in zip(preds_gen_reports_converted, preds_ref_reports_converted, CONDITIONS):
            if condition in five_conditions_to_evaluate:
                total_preds_gen_reports_5_conditions.extend(preds_gen_reports_condition)
                total_preds_ref_reports_5_conditions.extend(preds_ref_reports_condition)

            total_preds_gen_reports_14_conditions.extend(preds_gen_reports_condition)
            total_preds_ref_reports_14_conditions.extend(preds_ref_reports_condition)

            # compute and save scores for the given condition
            precision, recall, f1, _ = precision_recall_fscore_support(preds_ref_reports_condition, preds_gen_reports_condition, average="binary")
            acc = accuracy_score(preds_ref_reports_condition, preds_gen_reports_condition)

            language_model_scores["report"]["CE"][condition]["precision"] = precision
            language_model_scores["report"]["CE"][condition]["recall"] = recall
            language_model_scores["report"]["CE"][condition]["f1"] = f1
            language_model_scores["report"]["CE"][condition]["acc"] = acc

        # compute and save scores for all 14 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions)

        language_model_scores["report"]["CE"]["precision_micro_all"] = precision
        language_model_scores["report"]["CE"]["recall_micro_all"] = recall
        language_model_scores["report"]["CE"]["f1_micro_all"] = f1
        language_model_scores["report"]["CE"]["acc_all"] = acc

        # compute and save scores for the 5 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions)

        language_model_scores["report"]["CE"]["precision_micro_5"] = precision
        language_model_scores["report"]["CE"]["recall_micro_5"] = recall
        language_model_scores["report"]["CE"]["f1_micro_5"] = f1
        language_model_scores["report"]["CE"]["acc_5"] = acc

    def compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports):
        """
        example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        """
        preds_gen_reports_np = np.array(preds_gen_reports)  # array of shape (14 x num_reports), 14 for 14 conditions
        preds_ref_reports_np = np.array(preds_ref_reports)  # array of shape (14 x num_reports)
        preds_gen_reports_np = preds_gen_reports_np == 1
        preds_ref_reports_np = preds_ref_reports_np == 1

        tp = np.logical_and(preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fp = np.logical_and(preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fn = np.logical_and(~preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        tn = np.logical_and(~preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)

        # sum up the TP, FP, FN and TN for each report (i.e. for each column)
        tp_example = tp.sum(axis=0)  # int array of shape (num_reports)
        fp_example = fp.sum(axis=0)  # int array of shape (num_reports)
        fn_example = fn.sum(axis=0)  # int array of shape (num_reports)
        tn_example = tn.sum(axis=0)  # int array of shape (num_reports)

        # compute the scores for each report
        precision_example = tp_example / (tp_example + fp_example)  # float array of shape (num_reports)
        recall_example = tp_example / (tp_example + fn_example)  # float array of shape (num_reports)
        f1_example = (2 * tp_example) / (2 * tp_example + fp_example + fn_example)  # float array of shape (num_reports)
        acc_example = (tp_example + tn_example) / (tp_example + tn_example + fp_example + fn_example)  # float array of shape (num_reports)

        precision_example[np.isnan(precision_example)] = 0.0
        recall_example[np.isnan(recall_example)] = 0.0
        f1_example[np.isnan(f1_example)] = 0.0
        acc_example[np.isnan(acc_example)] = 0.0

        precision_example = float(precision_example.mean())
        recall_example = float(recall_example.mean())
        f1_example = float(f1_example.mean())
        acc_example = float(acc_example.mean())

        language_model_scores["report"]["CE"]["precision_example_all"] = precision_example
        language_model_scores["report"]["CE"]["recall_example_all"] = recall_example
        language_model_scores["report"]["CE"]["f1_example_all"] = f1_example
        language_model_scores["report"]["CE"]["acc_example_all"] = acc_example

    chexbert = get_chexbert()
    preds_gen_reports, preds_ref_reports = get_chexbert_labels_for_gen_and_ref_reports()

    compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports)
    compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports)


def compute_language_model_scores(gen_and_ref_sentences, gen_and_ref_reports):

    def compute_sentence_level_scores():
        def remove_gen_sents_corresponding_to_empty_ref_sents(gen_sents, ref_sents):
            filtered_gen_sents = []
            filtered_ref_sents = []

            for gen_sent, ref_sent in zip(gen_sents, ref_sents):
                if ref_sent != "":
                    filtered_gen_sents.append(gen_sent)
                    filtered_ref_sents.append(ref_sent)

            return filtered_gen_sents, filtered_ref_sents

        def compute_sent_level_scores_for_subset(subset, gen_sents, ref_sents):
            nlg_metrics = ["meteor"]
            nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents, ref_sents)
            meteor_score = nlg_scores["meteor"]
            language_model_scores[subset]["meteor"] = meteor_score

        def compute_sent_level_scores_for_region(region_name, gen_sents, ref_sents):
            nlg_metrics = ["meteor"]
            nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents, ref_sents)
            meteor_score = nlg_scores["meteor"]
            language_model_scores["region"][region_name]["meteor"] = meteor_score

        def compute_sent_level_meteor_ratio_score(gen_sents, ref_sents):
            gen_sents_for_computing_meteor_ratio_score = []
            ref_sents_for_computing_meteor_ratio_score = []

            num_generated_sentences_per_image = gen_and_ref_sentences["num_generated_sentences_per_image"]

            curr_index = 0
            for num_gen_sents in num_generated_sentences_per_image:
                gen_sents_single_image = gen_sents[curr_index:curr_index + num_gen_sents]
                ref_sents_single_image = ref_sents[curr_index:curr_index + num_gen_sents]

                curr_index += num_gen_sents

                gen_sents_single_image_filtered, ref_sents_single_image_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(gen_sents_single_image, ref_sents_single_image)

                for i, gen_sent in enumerate(gen_sents_single_image_filtered):
                    for j, ref_sent in enumerate(ref_sents_single_image_filtered):
                        if i == j:
                            continue  # skip "correct" match
                        else:
                            gen_sents_for_computing_meteor_ratio_score.append(gen_sent)
                            ref_sents_for_computing_meteor_ratio_score.append(ref_sent)

            nlg_metrics = ["meteor"]
            nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents_for_computing_meteor_ratio_score, ref_sents_for_computing_meteor_ratio_score)
            denominator_meteor_score = nlg_scores["meteor"]

            numerator_meteor_score = language_model_scores["all"]["meteor"]

            language_model_scores["all"]["meteor_ratio"] = numerator_meteor_score / denominator_meteor_score

        generated_sents = gen_and_ref_sentences["generated_sentences"]
        generated_sents_normal = gen_and_ref_sentences["generated_sentences_normal_selected_regions"]
        generated_sents_abnormal = gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"]

        reference_sents = gen_and_ref_sentences["reference_sentences"]
        reference_sents_normal = gen_and_ref_sentences["reference_sentences_normal_selected_regions"]
        reference_sents_abnormal = gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"]

        gen_sents_filtered, ref_sents_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents, reference_sents)
        gen_sents_normal_filtered, ref_sents_normal_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents_normal, reference_sents_normal)
        gen_sents_abnormal_filtered, ref_sents_abnormal_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents_abnormal, reference_sents_abnormal)

        compute_sent_level_scores_for_subset("all", gen_sents_filtered, ref_sents_filtered)
        compute_sent_level_scores_for_subset("normal", gen_sents_normal_filtered, ref_sents_normal_filtered)
        compute_sent_level_scores_for_subset("abnormal", gen_sents_abnormal_filtered, ref_sents_abnormal_filtered)

        compute_sent_level_meteor_ratio_score(generated_sents, reference_sents)

        for region_index, region_name in enumerate(ANATOMICAL_REGIONS):
            region_generated_sentences = gen_and_ref_sentences[region_index]["generated_sentences"]
            region_reference_sentences = gen_and_ref_sentences[region_index]["reference_sentences"]

            region_gen_sents_filtered, region_ref_sents_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(region_generated_sentences, region_reference_sentences)

            if len(region_gen_sents_filtered) != 0:
                compute_sent_level_scores_for_region(region_name, region_gen_sents_filtered, region_ref_sents_filtered)
            else:
                language_model_scores["region"][region_name]["meteor"] = -1

    def compute_report_level_scores():
        gen_reports = gen_and_ref_reports["generated_reports"]
        ref_reports = gen_and_ref_reports["reference_reports"]

        nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
        nlg_scores = compute_NLG_scores(nlg_metrics, gen_reports, ref_reports)

        for nlg_metric_name, score in nlg_scores.items():
            language_model_scores["report"][nlg_metric_name] = score

        compute_clinical_efficacy_scores(language_model_scores, gen_reports, ref_reports)

    def create_language_model_scores_dict():
        language_model_scores = {}
        # BLEU 1-4
        # METEOR
        # ROUGE-L
        # Cider-D
        # CE scores (P, R, F1, acc)
        language_model_scores["report"] = {f"bleu_{i}": None for i in range(1, 5)}
        language_model_scores["report"]["meteor"] = None
        language_model_scores["report"]["rouge"] = None
        language_model_scores["report"]["cider"] = None
        language_model_scores["report"]["CE"] = {
            "precision_micro_5": None,
            "recall_micro_5": None,
            "f1_micro_5": None,
            "acc_5": None,
            "precision_micro_all": None,
            "recall_micro_all": None,
            "acc_all": None
        }

        for condition in CONDITIONS:
            language_model_scores["report"]["CE"][condition] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "acc": None
            }

        # following Nicolson (https://arxiv.org/pdf/2201.09405.pdf), we evaluate the example-based CE scores over all conditions
        # example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        language_model_scores["report"]["CE"]["precision_example_all"] = None
        language_model_scores["report"]["CE"]["recall_example_all"] = None
        language_model_scores["report"]["CE"]["f1_example_all"] = None
        language_model_scores["report"]["CE"]["acc_example_all"] = None

        for subset in ["all", "normal", "abnormal"]:
            language_model_scores[subset] = {"meteor": None}

        # we also compute these scores for each region individually
        language_model_scores["region"] = {}
        for region_name in ANATOMICAL_REGIONS:
            language_model_scores["region"][region_name] = {"meteor": None}

        language_model_scores["all"]["meteor_ratio"] = None

        return language_model_scores

    language_model_scores = create_language_model_scores_dict()

    compute_report_level_scores()
    compute_sentence_level_scores()

    return language_model_scores


def write_sentences_and_reports_to_file(
    gen_and_ref_sentences,
    gen_and_ref_reports,
    gen_sentences_with_corresponding_regions,
    generated_sentences_and_reports_folder_path,
    overall_steps_taken,
):
    def write_sentences():
        txt_file_name = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences", f"generated_sentences_step_{overall_steps_taken}")
        txt_file_name_abnormal = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences", f"generated_abnormal_sentences_step_{overall_steps_taken}")

        with open(txt_file_name, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

        with open(txt_file_name_abnormal, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences_abnormal_regions, reference_sentences_abnormal_regions):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

    def write_reports():
        txt_file_name = os.path.join(
            generated_sentences_and_reports_folder_path,
            "generated_reports",
            f"generated_reports_step_{overall_steps_taken}",
        )

        with open(txt_file_name, "w") as f:
            for gen_report, ref_report, removed_similar_gen_sents, gen_sents_with_regions_single_report in zip(
                generated_reports,
                reference_reports,
                removed_similar_generated_sentences,
                gen_sentences_with_corresponding_regions
            ):
                f.write(f"Generated report: {gen_report}\n\n")
                f.write(f"Reference report: {ref_report}\n\n")

                f.write("Generated sentences with their regions:\n")
                for region_name, gen_sent in gen_sents_with_regions_single_report:
                    f.write(f"\t{region_name}: {gen_sent}\n")
                f.write("\n")

                f.write("Generated sentences that were removed:\n")
                for gen_sent, list_similar_gen_sents in removed_similar_gen_sents.items():
                    f.write(f"\t{gen_sent} == {list_similar_gen_sents}\n")
                f.write("\n")

                f.write("=" * 30)
                f.write("\n\n")

    num_generated_sentences_to_save = NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE * BATCH_SIZE
    num_generated_reports_to_save = NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE

    generated_sentences = gen_and_ref_sentences["generated_sentences"][:num_generated_sentences_to_save]
    generated_sentences_abnormal_regions = gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"][:num_generated_sentences_to_save]
    reference_sentences = gen_and_ref_sentences["reference_sentences"][:num_generated_sentences_to_save]
    reference_sentences_abnormal_regions = gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"][:num_generated_sentences_to_save]

    write_sentences()

    generated_reports = gen_and_ref_reports["generated_reports"][:num_generated_reports_to_save]
    reference_reports = gen_and_ref_reports["reference_reports"][:num_generated_reports_to_save]
    removed_similar_generated_sentences = gen_and_ref_reports["removed_similar_generated_sentences"][:num_generated_reports_to_save]

    write_reports()


def get_plot_title(region_set, region_indices, region_colors, class_detected_img) -> str:
    class_detected = [class_detected_img[region_index] for region_index in region_indices]

    region_set = [
        region + f" ({color})" if cls_detect else region + f" ({color}, nd)"
        for region, color, cls_detect in zip(region_set, region_colors, class_detected)
    ]

    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


def get_generated_sentence_for_region(
    generated_sentences_for_selected_regions, selected_regions, num_img, region_index
) -> str:
    selected_regions_flat = selected_regions.reshape(-1)
    cum_sum_true_values = np.cumsum(selected_regions_flat)

    cum_sum_true_values = cum_sum_true_values.reshape(selected_regions.shape)
    cum_sum_true_values -= 1

    index = cum_sum_true_values[num_img][region_index]

    return generated_sentences_for_selected_regions[index]


def transform_sentence_to_fit_under_image(sentence):
    max_line_length = 60
    if len(sentence) < max_line_length:
        return sentence

    words = sentence.split()
    transformed_sent = ""
    current_line_length = 0
    prefix_for_alignment = "\n" + " " * 20
    for word in words:
        if len(word) + current_line_length > max_line_length:
            word = f"{prefix_for_alignment}{word}"
            current_line_length = -len(prefix_for_alignment)

        current_line_length += len(word)
        transformed_sent += word + " "

    return transformed_sent


def update_region_set_text(
    region_set_text,
    color,
    reference_sentences_img,
    generated_sentences_for_selected_regions,
    region_index,
    selected_regions,
    num_img,
):
    region_set_text += f"({color}):\n"
    reference_sentence_region = reference_sentences_img[region_index]
    reference_sentence_region = transform_sentence_to_fit_under_image(reference_sentence_region)

    region_set_text += f"  reference: {reference_sentence_region}\n"

    box_region_selected = selected_regions[num_img][region_index]
    if not box_region_selected:
        region_set_text += "  generated: [REGION NOT SELECTED]\n\n"
    else:
        generated_sentence_region = get_generated_sentence_for_region(
            generated_sentences_for_selected_regions, selected_regions, num_img, region_index
        )
        generated_sentence_region = transform_sentence_to_fit_under_image(generated_sentence_region)
        region_set_text += f"  generated: {generated_sentence_region}\n\n"

    return region_set_text


def plot_box(box, ax, clr, linestyle, region_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(
        plt.Rectangle(xy=(x0, y0), height=h, width=w, fill=False, color=clr, linewidth=1, linestyle=linestyle)
    )

    if not region_detected:
        ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)


def plot_detections_and_sentences_to_tensorboard(
    writer,
    num_batch,
    overall_steps_taken,
    images,
    image_targets,
    selected_regions,
    detections,
    class_detected,
    reference_sentences,
    generated_sentences_for_selected_regions,
):
    pred_boxes_batch = detections["top_region_boxes"]

    gt_boxes_batch = torch.stack([t["boxes"] for t in image_targets], dim=0)

    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
    region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
    region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]

    images = images.numpy().transpose(0, 2, 3, 1)

    for num_img, image in enumerate(images):  # 遍历一个batch里面的所有图像

        gt_boxes_img = gt_boxes_batch[num_img]
        pred_boxes_img = pred_boxes_batch[num_img]
        class_detected_img = class_detected[num_img].tolist()
        reference_sentences_img = reference_sentences[num_img]

        for num_region_set, region_set in enumerate(regions_sets):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            plt.imshow(image, cmap="gray")
            plt.axis("on")

            region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]  # 部分解剖区域（6个）对应的编号list
            region_colors = ["b", "g", "r", "c", "m", "y"]

            if num_region_set == 4: # 最后一个region_set只有五个区域，只需要五个颜色
                region_colors.pop()

            region_set_text = ""

            for region_index, color in zip(region_indices, region_colors):
                # box_gt and box_pred are both [List[float]] of len 4
                box_gt = gt_boxes_img[region_index].tolist()
                box_pred = pred_boxes_img[region_index].tolist()
                box_region_detected = class_detected_img[region_index]  # bool type

                plot_box(box_gt, ax, clr=color, linestyle="solid", region_detected=box_region_detected)

                # only plot predicted box if class was actually detected
                if box_region_detected:
                    plot_box(box_pred, ax, clr=color, linestyle="dashed")

                region_set_text = update_region_set_text(
                    region_set_text,
                    color,
                    reference_sentences_img,
                    generated_sentences_for_selected_regions,
                    region_index,
                    selected_regions,
                    num_img,
                )

            title = get_plot_title(region_set, region_indices, region_colors, class_detected_img)
            ax.set_title(title)

            plt.xlabel(region_set_text, loc="left")
            buf = io.BytesIO()
            fig.savefig(buf, bbox_inches="tight")
            buf.seek(0)
            im = Image.open(buf)
            im = np.asarray(im)[..., :3]

            writer_image_num = num_batch * BATCH_SIZE + num_img
            writer.add_image(
                f"img_{writer_image_num}_region_set_{num_region_set}",
                im,
                global_step=overall_steps_taken,
                dataformats="HWC",
            )

            plt.close(fig)


def update_gen_sentences_with_corresponding_regions(
    gen_sentences_with_corresponding_regions,
    generated_sents_for_selected_regions,
    selected_regions
):
    def get_region_name(region_index: int):
        for i, region_name in enumerate(ANATOMICAL_REGIONS):
            if i == region_index:
                return region_name

    index_gen_sentence = 0

    # selected_regions_single_image is a row with 29 bool values corresponding to a single image
    for selected_regions_single_image in selected_regions:
        gen_sents_with_regions_single_image = []

        for region_index, region_selected_bool in enumerate(selected_regions_single_image):
            if region_selected_bool:
                region_name = get_region_name(region_index)
                gen_sent = generated_sents_for_selected_regions[index_gen_sentence]

                gen_sents_with_regions_single_image.append((region_name, gen_sent))

                index_gen_sentence += 1

        gen_sentences_with_corresponding_regions.append(gen_sents_with_regions_single_image)


def update_num_generated_sentences_per_image(
    gen_and_ref_sentences: dict,
    selected_regions: np.array
):
    num_gen_sents_per_image = selected_regions.sum(axis=1).tolist()  # indices is a list[int] of len(batch_size)
    gen_and_ref_sentences["num_generated_sentences_per_image"].extend(num_gen_sents_per_image)


def update_gen_and_ref_sentences_for_regions(
    gen_and_ref_sentences: dict,
    generated_sents_for_selected_regions: list[str],
    reference_sents_for_selected_regions: list[str],
    selected_regions: np.array
):
    index_gen_ref_sentence = 0

    # of shape (batch_size * 29)
    selected_regions_flat = selected_regions.reshape(-1)
    for curr_index, region_selected_bool in enumerate(selected_regions_flat):
        if region_selected_bool:
            region_index = curr_index % 29
            gen_sent = generated_sents_for_selected_regions[index_gen_ref_sentence]
            ref_sent = reference_sents_for_selected_regions[index_gen_ref_sentence]

            gen_and_ref_sentences[region_index]["generated_sentences"].append(gen_sent)
            gen_and_ref_sentences[region_index]["reference_sentences"].append(ref_sent)

            index_gen_ref_sentence += 1


def get_generated_reports(generated_sentences_for_selected_regions, selected_regions, sentence_tokenizer, bertscore_threshold):
    def remove_duplicate_generated_sentences(gen_report_single_image, bert_score):
        def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
            for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
                if gen_sent in lists_of_gen_sents_to_be_removed:
                    return True

            return False

        gen_sents_single_image = sentence_tokenizer(gen_report_single_image).sents

        gen_sents_single_image = [sent.text for sent in gen_sents_single_image]

        gen_sents_single_image = list(dict.fromkeys(gen_sents_single_image))

        similar_generated_sents_to_be_removed = defaultdict(list)

        for i in range(len(gen_sents_single_image)):
            gen_sent_1 = gen_sents_single_image[i]

            for j in range(i + 1, len(gen_sents_single_image)):
                if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                    break

                gen_sent_2 = gen_sents_single_image[j]
                if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                    continue

                # bert_score_result = bert_score_source.compute(
                #     lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
                # )
                # TODO adopt source code of bert_score_source to calculate rather than evaluate.load("bertscore")
                bert_score_result = bert_score(
                    [gen_sent_1], [gen_sent_2], lang="en", model_type="distilbert-base-uncased"
                )

                # print(bert_score_result, type(bert_score_result))
                P, R, F1 = bert_score_result
                # print(P, R, F1)

                # if bert_score_result["f1"][0] > bertscore_threshold:  # 初始版本（使用evaluate.load("bertscore")）
                if F1 > bertscore_threshold:
                    # remove the generated similar sentence that is shorter
                    if len(gen_sent_1) > len(gen_sent_2):
                        similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                    else:
                        similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

        gen_report_single_image = " ".join(
            sent for sent in gen_sents_single_image if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
        )

        return gen_report_single_image, similar_generated_sents_to_be_removed

    # bert_score_source = evaluate.load("bertscore")

    # import bert_score_source.bert_score_source.score as bertscore
    # bert_score_source = evaluate.load("/public/home/zhangke/rgrg-main/src/full_model/evaluate_full_model/bertscore.py")

    generated_reports = []
    removed_similar_generated_sentences = []
    curr_index = 0

    for selected_regions_single_image in selected_regions:
        num_selected_regions_single_image = np.sum(selected_regions_single_image)

        gen_sents_single_image = generated_sentences_for_selected_regions[
            curr_index: curr_index + num_selected_regions_single_image
        ]

        curr_index += num_selected_regions_single_image

        gen_report_single_image = " ".join(sent for sent in gen_sents_single_image)

        gen_report_single_image, similar_generated_sents_to_be_removed = remove_duplicate_generated_sentences(
            gen_report_single_image, bert_score
        )

        generated_reports.append(gen_report_single_image)
        removed_similar_generated_sentences.append(similar_generated_sents_to_be_removed)

    return generated_reports, removed_similar_generated_sentences


def get_ref_sentences_for_selected_regions(reference_sentences, selected_regions):
    reference_sentences = np.asarray(reference_sentences)

    ref_sentences_for_selected_regions = reference_sentences[selected_regions]

    return ref_sentences_for_selected_regions.tolist()


def get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sentences_for_selected_regions, reference_sentences_for_selected_regions):
    selected_region_is_abnormal = region_is_abnormal[selected_regions]

    gen_sents_for_selected_regions = np.asarray(generated_sentences_for_selected_regions)
    ref_sents_for_selected_regions = np.asarray(reference_sentences_for_selected_regions)

    gen_sents_for_normal_selected_regions = gen_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
    gen_sents_for_abnormal_selected_regions = gen_sents_for_selected_regions[selected_region_is_abnormal].tolist()

    ref_sents_for_normal_selected_regions = ref_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
    ref_sents_for_abnormal_selected_regions = ref_sents_for_selected_regions[selected_region_is_abnormal].tolist()

    return (
        gen_sents_for_normal_selected_regions,
        gen_sents_for_abnormal_selected_regions,
        ref_sents_for_normal_selected_regions,
        ref_sents_for_abnormal_selected_regions,
    )


def evaluate_language_model(model, val_dl, tokenizer, writer, run_params, generated_sentences_and_reports_folder_path):
    epoch = run_params["epoch"]
    overall_steps_taken = run_params["overall_steps_taken"]
    log_file = run_params["log_file"]

    gen_and_ref_sentences = {
        "generated_sentences": [],
        "generated_sentences_normal_selected_regions": [],
        "generated_sentences_abnormal_selected_regions": [],
        "reference_sentences": [],
        "reference_sentences_normal_selected_regions": [],
        "reference_sentences_abnormal_selected_regions": [],
        "num_generated_sentences_per_image": []
    }

    for region_index, _ in enumerate(ANATOMICAL_REGIONS):
        gen_and_ref_sentences[region_index] = {
            "generated_sentences": [],
            "reference_sentences": []
        }

    gen_and_ref_reports = {
        "generated_reports": [],
        "removed_similar_generated_sentences": [],
        "reference_reports": [],
    }

    gen_sentences_with_corresponding_regions = []

    num_batches_to_process_for_image_plotting = NUM_IMAGES_TO_PLOT // BATCH_SIZE

    oom = False

    sentence_tokenizer = spacy.load("en_core_web_trf")

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION):
            if num_batch >= NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION:
                break

            images = batch["images"]
            image_targets = batch["image_targets"]
            region_is_abnormal = batch["region_is_abnormal"].numpy()

            reference_sentences = batch["reference_sentences"]

            reference_reports = batch["reference_reports"]

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model.generate(
                        images.to(device, non_blocking=True),
                        max_length=MAX_NUM_TOKENS_GENERATE,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True

                    with open(log_file, "a") as f:
                        f.write("Generation:\n")
                        f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
                        f.write(f"Error message: {str(e)}\n\n")
                else:
                    raise e

            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False
                continue

            if output == -1:
                with open(log_file, "a") as f:
                    f.write("Generation:\n")
                    f.write(f"Empty region features before language model at epoch {epoch}, batch number {num_batch}.\n\n")

                continue
            else:
                beam_search_output, selected_regions, detections, class_detected = output
                selected_regions = selected_regions.detach().cpu().numpy()

            generated_sents_for_selected_regions = tokenizer.batch_decode(
                beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            reference_sents_for_selected_regions = get_ref_sentences_for_selected_regions(
                reference_sentences, selected_regions
            )

            (
                gen_sents_for_normal_selected_regions,
                gen_sents_for_abnormal_selected_regions,
                ref_sents_for_normal_selected_regions,
                ref_sents_for_abnormal_selected_regions,
            ) = get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sents_for_selected_regions, reference_sents_for_selected_regions)

            # 超级费时间(CPU)
            generated_reports, removed_similar_generated_sentences = get_generated_reports(
                generated_sents_for_selected_regions,
                selected_regions,
                sentence_tokenizer,
                BERTSCORE_SIMILARITY_THRESHOLD
            )

            gen_and_ref_sentences["generated_sentences"].extend(generated_sents_for_selected_regions)
            gen_and_ref_sentences["generated_sentences_normal_selected_regions"].extend(gen_sents_for_normal_selected_regions)
            gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"].extend(gen_sents_for_abnormal_selected_regions)
            gen_and_ref_sentences["reference_sentences"].extend(reference_sents_for_selected_regions)
            gen_and_ref_sentences["reference_sentences_normal_selected_regions"].extend(ref_sents_for_normal_selected_regions)
            gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"].extend(ref_sents_for_abnormal_selected_regions)
            gen_and_ref_reports["generated_reports"].extend(generated_reports)
            gen_and_ref_reports["reference_reports"].extend(reference_reports)
            gen_and_ref_reports["removed_similar_generated_sentences"].extend(removed_similar_generated_sentences)

            update_gen_and_ref_sentences_for_regions(gen_and_ref_sentences, generated_sents_for_selected_regions, reference_sents_for_selected_regions, selected_regions)
            update_num_generated_sentences_per_image(gen_and_ref_sentences, selected_regions)

            if num_batch < NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE:
                update_gen_sentences_with_corresponding_regions(gen_sentences_with_corresponding_regions, generated_sents_for_selected_regions, selected_regions)

            if num_batch < num_batches_to_process_for_image_plotting:  # 选取一定阈值的数量进行绘制
                plot_detections_and_sentences_to_tensorboard(          # 绘制检测框和句子
                    writer,
                    num_batch,
                    overall_steps_taken,
                    images,
                    image_targets,
                    selected_regions,
                    detections,
                    class_detected,
                    reference_sentences,
                    generated_sents_for_selected_regions,
                )

    write_sentences_and_reports_to_file(
        gen_and_ref_sentences,
        gen_and_ref_reports,
        gen_sentences_with_corresponding_regions,
        generated_sentences_and_reports_folder_path,
        overall_steps_taken,
    )

    language_model_scores = compute_language_model_scores(gen_and_ref_sentences, gen_and_ref_reports)

    return language_model_scores
