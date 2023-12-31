from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.binary_classifier.binary_classifier_region_abnormal import BinaryClassifierRegionAbnormal
from src.binary_classifier.binary_classifier_region_selection import BinaryClassifierRegionSelection
from src.object_detector.object_detector import ObjectDetector
from src.language_model.language_model import LanguageModel

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.data as data
from torch.nn.utils import weight_norm
from tqdm import tqdm
from utils import cofig
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.layers import smooth_l1_loss


class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)
        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.activate is not None:
            x = self.ac_fn(x)
        return x


class ApplyAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(ApplyAttention, self).__init__()
        self.glimpses = glimpses
        layers = []
        for g in range(self.glimpses):
            layers.append(ApplySingleAttention(v_features, q_features, mid_features, drop))
        self.glimpse_layers = nn.ModuleList(layers)

    def forward(self, v, q, atten):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        atten:  batch x glimpses x v_num x q_num
        """
        for g in range(self.glimpses):
            atten_h = self.glimpse_layers[g](v, q, atten)
            q = q + atten_h
        # q = q * q_mask.unsqueeze(2)
        return q.sum(1)


class ApplySingleAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features, activate='relu', drop=drop)
        self.lin_q = FCNet(q_features, mid_features, activate='relu', drop=drop)
        self.lin_atten = FCNet(mid_features, mid_features, drop=drop)

    def forward(self, v, q, atten):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        atten:  batch x v_num x q_num
        """

        # apply single glimpse attention
        v_ = self.lin_v(v).transpose(1, 2).unsqueeze(2)  # batch, dim, 1, num_obj
        q_ = self.lin_q(q).transpose(1, 2).unsqueeze(3)  # batch, dim, que_len, 1
        # v_ = torch.matmul(v_, atten.unsqueeze(1)) # batch, dim, 1, que_len
        # This is the only way I found to make it match the dimension in the previous comment: # batch, dim, 1, que_len
        v_ = torch.matmul(v_.squeeze(2), atten.transpose(3, 1).squeeze(2)).unsqueeze(2)
        h_ = torch.matmul(v_, q_)  # batch, dim, 1, 1
        h_ = h_.squeeze(3).squeeze(2)  # batch, dim

        atten_h = self.lin_atten(h_.unsqueeze(1))

        return atten_h


class IterativeSceneGraphGeneration(nn.Module):
    def __init__(self, img_num_obj=151, img_num_rel=51, txt_num_obj=4460, txt_num_rel=646):
        super(IterativeSceneGraphGeneration, self).__init__()
        self.embed_dim = 512
        self.hidden_dim = 512
        self.final_dim = 1024
        self.num_layer = 2
        self.margin = 1.0
        self.img_num_obj = img_num_obj
        self.img_num_rel = img_num_rel
        self.txt_num_obj = txt_num_obj
        self.txt_num_rel = txt_num_rel
        self.attrProto = config.attrProto
        self.tau = config.tau
        self.cosine = config.consine
        self.img_obj_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_head_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_tail_embed = nn.Embedding(self.img_num_obj, self.embed_dim)
        self.img_rel_pred_embed = nn.Embedding(self.img_num_rel, self.embed_dim)
        self.txt_obj_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_head_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_tail_embed = nn.Embedding(self.txt_num_obj, self.embed_dim)
        self.txt_rel_pred_embed = nn.Embedding(self.txt_num_rel, self.embed_dim)

        self.apply_attention = ApplyAttention(
            v_features=self.embed_dim * 3,
            q_features=self.embed_dim,
            mid_features=self.hidden_dim,
            glimpses=self.num_layer,
            drop=0.2, )

        self.final_fc = nn.Sequential(*[nn.Linear(self.hidden_dim, self.hidden_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hidden_dim, self.final_dim),
                                        nn.ReLU(inplace=True)
                                        ])

    def encode(self, inp_dict, is_img=False, is_txt=False):
        assert is_img + is_txt
        if len(inp_dict['relations'].shape) == 1:
            inp_dict['relations'] = torch.zeros(1, 3).to(inp_dict['entities'].device).long()
            inp_dict['graph'] = torch.zeros(len(inp_dict['entities']), 1).to(inp_dict['entities'].device).float()

        if is_img:
            obj_encode = self.img_obj_embed(inp_dict['entities'])
            rel_head_encode = self.img_rel_head_embed(inp_dict['relations'][:, 0])
            rel_tail_encode = self.img_rel_tail_embed(inp_dict['relations'][:, 1])
            rel_pred_encode = self.img_rel_pred_embed(inp_dict['relations'][:, 2])
        elif is_txt:
            obj_encode = self.txt_obj_embed(inp_dict['entities'])
            rel_head_encode = self.txt_rel_head_embed(inp_dict['relations'][:, 0])
            rel_tail_encode = self.txt_rel_tail_embed(inp_dict['relations'][:, 1])
            rel_pred_encode = self.txt_rel_pred_embed(inp_dict['relations'][:, 2])
        else:
            print('ERROR')

        rel_encode = torch.cat((rel_head_encode, rel_tail_encode, rel_pred_encode), dim=-1)

        atten = inp_dict['graph'].transpose(0, 1)  # num_rel, num_obj
        atten = atten / (atten.sum(0).view(1, -1) + 1e-9)

        sg_encode = self.apply_attention(rel_encode.unsqueeze(0), obj_encode.unsqueeze(0), atten.unsqueeze(0))

        return self.final_fc(sg_encode).sum(0).view(1, -1)

    def forward(self, top_region_features):
        loss = []
        sgfeat = []
        fg_imgs = []
        fg_txts = []
        bg_imgs = []
        bg_txts = []
        is_test = False
        for i in range(len(top_region_features)):
            for j in range(i+1, len(top_region_features)):
                h_max = 0
                k_max = 0
                for k in range(self.attrProto):
                    C = ApplyAttention(top_region_features[i],top_region_features[j],top_region_features[j])
                    h = self.cosine(C, self.attrProto[k])
                    if h > h_max:
                        h_max = h
                        k_max = k
                if h_max > self.tau:
                    fg_imgs.append(top_region_features[i])
                    bg_imgs.append(top_region_features[j])
                    fg_txts.append(self.attrProto[k_max])
                    bg_txts.append(self.attrProto[k_max])
        for fg_img, fg_txt, bg_img, bg_txt in zip(fg_imgs, fg_txts, bg_imgs, bg_txts):
            fg_img_encode = self.encode(fg_img, is_img=True)
            fg_txt_encode = self.encode(fg_txt, is_txt=True)
            bg_img_encode = self.encode(bg_img, is_img=True)
            bg_txt_encode = self.encode(bg_txt, is_txt=True)

            fg_intra = smooth_l1_loss(fg_img_encode, fg_txt_encode)
            fg_inter = smooth_l1_loss(fg_img_encode, bg_txt_encode)
            triplet_fg = fg_intra + self.margin - fg_inter
            triplet_fg = triplet_fg * (triplet_fg >= 0).float()
            loss.append(triplet_fg.sum())

            bg_intra = smooth_l1_loss(bg_txt_encode, bg_img_encode)
            bg_inter = smooth_l1_loss(fg_txt_encode, bg_img_encode)
            triplet_bg = bg_intra + self.margin - bg_inter
            triplet_bg = triplet_bg * (triplet_bg >= 0).float()

            loss.append(triplet_bg.sum())
            sgfeat.append([fg_img_encode, fg_txt_encode])


        return loss, sgfeat



class ReportGenerationModel(nn.Module):

    def __init__(self, pretrain_without_lm_model=False):
        super().__init__()
        self.pretrain_without_lm_model = pretrain_without_lm_model

        self.object_detector = ObjectDetector(return_feature_vectors=True)
        # Load the best object detector from the 1st training stage here when starting the 2nd training stage
        # path_to_best_object_detector_weights = "/u/home/tanida/runs/object_detector/run_10/weights/val_loss_13.482_epoch_6.pth"
        # self.object_detector.load_state_dict(torch.load(path_to_best_object_detector_weights))

        self.binary_classifier_region_selection = BinaryClassifierRegionSelection()
        self.binary_classifier_region_abnormal = BinaryClassifierRegionAbnormal()

        self.iter_SGG = IterativeSceneGraphGeneration()

        self.language_model = LanguageModel()

    def forward(
        self,
        images: torch.FloatTensor,  # images is of shape [batch_size x 1 x 512 x 512] (whole gray-scale images of size 512 x 512)
        image_targets: List[Dict],  # contains a dict for every image with keys "boxes" and "labels"
        input_ids: torch.LongTensor,  # shape [(batch_size * 29) x seq_len], 1 sentence for every region for every image (sentence can be empty, i.e. "")
        attention_mask: torch.FloatTensor,  # shape [(batch_size * 29) x seq_len]
        region_has_sentence: torch.BoolTensor,  # shape [batch_size x 29], ground truth boolean mask that indicates if a region has a sentence or not
        region_is_abnormal: torch.BoolTensor,  # shape [batch_size x 29], ground truth boolean mask that indicates if a region has is abnormal or not
        return_loss: bool = True,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
    ):
        """
        Forward method is used for training and evaluation of model.
        Generate method is used for inference.
        """
        if self.training:
            # top_region_features of shape [batch_size x 29 x 1024] (i.e. 1 feature vector for every region for every image in batch)
            # class_detected is a boolean tensor of shape [batch_size x 29]. Its value is True for a class if the object detector detected the class/region in the image
            obj_detector_loss_dict, top_region_features, class_detected = self.object_detector(images, image_targets)

            # delete tensors that we don't need anymore to free up GPU resources
            del images
            del image_targets

            # during training, only get the two losses for the two binary classifiers

            classifier_loss_region_selection = self.binary_classifier_region_selection(
                top_region_features, class_detected, return_loss=True, region_has_sentence=region_has_sentence
            )

            classifier_loss_region_abnormal = self.binary_classifier_region_abnormal(
                top_region_features, class_detected, region_is_abnormal
            )

            if self.pretrain_without_lm_model:
                return obj_detector_loss_dict, classifier_loss_region_selection, classifier_loss_region_abnormal

            # to train the decoder, we want to use only the top region features (and corresponding input_ids, attention_mask)
            # of regions that were both detected by the object detector and have a sentence as the ground truth
            # this is done under the assumption that at inference time, the binary classifier for region selection will do an adequate job
            # at selecting those regions that need a sentence to be generated by itself
            valid_input_ids, valid_attention_mask, valid_region_features = self.get_valid_decoder_input_for_training(
                class_detected, region_has_sentence, input_ids, attention_mask, top_region_features
            )

            del top_region_features
            del region_has_sentence
            del region_is_abnormal
            del class_detected
            del input_ids
            del attention_mask

        else:
            # during evaluation, also return detections (i.e. detected bboxes)
            obj_detector_loss_dict, detections, top_region_features, class_detected = self.object_detector(images, image_targets)

            del images
            del image_targets

            # during evaluation, for the binary classifier for region selection, get the loss, the regions that were selected by the classifier
            # (and that were also detected) and the corresponding region features (selected_region_features)
            # this is done to evaluate the decoder under "real-word" conditions, i.e. the binary classifier decides which regions get a sentence
            classifier_loss_region_selection, selected_regions, selected_region_features = self.binary_classifier_region_selection(
                top_region_features, class_detected, return_loss=True, region_has_sentence=region_has_sentence
            )

            # for the binary classifier for abnormal/normal detection, get the loss and the predicted abnormal regions
            classifier_loss_region_abnormal, predicted_abnormal_regions = self.binary_classifier_region_abnormal(
                top_region_features, class_detected, region_is_abnormal
            )

            if self.pretrain_without_lm_model:
                return obj_detector_loss_dict, classifier_loss_region_selection, classifier_loss_region_abnormal, detections, class_detected, selected_regions, predicted_abnormal_regions

            del top_region_features
            del region_has_sentence
            del region_is_abnormal

            # use the selected_regions mask to filter the inputs_ids and attention_mask to those that correspond to regions that were selected
            valid_input_ids, valid_attention_mask = self.get_valid_decoder_input_for_evaluation(selected_regions, input_ids, attention_mask)
            valid_region_features = selected_region_features

            del input_ids
            del attention_mask

        # valid_input_ids can be empty if during:
        # training:
        #   - the regions that have a gt sentence (specified by region_has_sentence) were all not detected (specified by class_detected).
        #   This can happend if e.g. a lateral chest x-ray was erroneously included in the dataset (and hence the object detector not detecting
        #   any regions, since it was trained on frontal chest x-rays)
        # evaluation:
        #   - no regions were selected by the binary classifier (specified by selected_regions)
        #   - the regions that were selected by the binary classifier for region selection were all not detected (also specified by selected_regions,
        #   since class_detected is encoded in selected_regions). Again, the reason might be a bad input image
        #
        # empty valid_input_ids (and thus empty valid_attention_mask, valid_region_features) will throw an exception in the language model,
        # which is why we have to return early
        if valid_input_ids.shape[0] == 0:
            return -1

        language_model_loss = self.language_model(
            valid_input_ids,
            valid_attention_mask,
            valid_region_features,
            return_loss,
            past_key_values,
            position_ids,
            use_cache,
        )

        del valid_input_ids
        del valid_attention_mask
        del valid_region_features

        if self.training:
            return obj_detector_loss_dict, classifier_loss_region_selection, classifier_loss_region_abnormal, language_model_loss
        else:
            # class_detected needed to evaluate how good the object detector is at detecting the different regions during evaluation
            # detections and class_detected needed to compute IoU of object detector during evaluation
            # selected_regions needed to evaluate binary classifier for region selection during evaluation and
            # to map each generated sentence to its corresponding region (for example for plotting)
            # predicted_abnormal_regions needed to evalute the binary classifier for normal/abnormal detection
            return (
                obj_detector_loss_dict,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                language_model_loss,
                detections,
                class_detected,
                selected_regions,
                predicted_abnormal_regions
            )

    def get_valid_decoder_input_for_training(
        self,
        class_detected,  # shape [batch_size x 29]
        region_has_sentence,  # shape [batch_size x 29]
        input_ids,  # shape [(batch_size * 29) x seq_len]
        attention_mask,  # shape [(batch_size * 29) x seq_len]
        region_features,  # shape [batch_size x 29 x 1024]
    ):
        """
        We want to train the decoder only on region features (and corresponding input_ids/attention_mask) whose corresponding sentences are non-empty and
        that were detected by the object detector.
        """
        # valid is of shape [batch_size x 29]
        valid = torch.logical_and(class_detected, region_has_sentence)

        # reshape to [(batch_size * 29)], such that we can apply the mask to input_ids and attention_mask
        valid_reshaped = valid.reshape(-1)

        valid_input_ids = input_ids[valid_reshaped]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x seq_len]
        valid_attention_mask = attention_mask[valid_reshaped]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x seq_len]
        valid_region_features = region_features[valid]  # of shape [num_detected_regions_with_non_empty_gt_phrase_in_batch x 1024]

        return valid_input_ids, valid_attention_mask, valid_region_features

    def get_valid_decoder_input_for_evaluation(
        self,
        selected_regions,  # shape [batch_size x 29]
        input_ids,  # shape
            # [(batch_size * 29) x seq_len]
        attention_mask  # shape [(batch_size * 29) x seq_len]
    ):
        """
        For evaluation, we want to evaluate the decoder on the top_region_features selected by the classifier to get a sentence generated.
        We also have to get the corresponding input_ids and attention_mask accordingly.
        """
        # reshape to [(batch_size * 29)]
        selected_regions = selected_regions.reshape(-1)

        valid_input_ids = input_ids[selected_regions]  # of shape [num_regions_selected_in_batch x seq_len]
        valid_attention_mask = attention_mask[selected_regions]  # of shape [num_regions_selected_in_batch x seq_len]

        return valid_input_ids, valid_attention_mask

    @torch.no_grad()
    def generate(
        self,
        images: torch.FloatTensor,  # images is of shape [batch_size x 1 x 512 x 512] (whole gray-scale images of size 512 x 512)
        max_length: int = None,
        num_beams: int = 1,
        num_beam_groups: int = 1,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        early_stopping: bool = False,
    ):
        # top_region_features of shape [batch_size x 29 x 1024]
        _, detections, top_region_features, class_detected = self.object_detector(images)

        del images

        _, scene_graph_features = self.iter_SGG(top_region_features)

        # selected_region_features is of shape [num_regions_selected_in_batch x 1024]
        # selected_regions is of shape [batch_size x 29] and is True for regions that should get a sentence
        # (it has exactly num_regions_selected_in_batch True values)
        selected_regions, selected_region_features = self.binary_classifier_region_selection(
            top_region_features, class_detected, return_loss=False
        )

        del top_region_features

        # selected_region_features can be empty if no region was both detected by the object detector and selected
        # by the binary classifier to get a sentence generated. This can happen especially early on in training
        # Since this would throw an exception in the language model, we return early
        if selected_region_features.shape[0] == 0:
            return -1

        # output_ids of shape (num_regions_selected_in_batch x longest_generated_sequence_length)
        output_ids = self.language_model.generate(
            selected_region_features,
            scene_graph_features,
            max_length,
            num_beams,
            num_beam_groups,
            do_sample,
            num_return_sequences,
            early_stopping,
        )

        del selected_region_features

        return output_ids, selected_regions, detections, class_detected
