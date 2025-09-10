# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
VABooster model and criterion classes.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from vabooster.span_utils import generalized_temporal_iou, span_cxw_to_xx

from vabooster.matcher import build_matcher
from vabooster.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from vabooster.position_encoding import build_position_encoding
from vabooster.misc import accuracy
import numpy as np
import copy

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start+len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start

def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a==b:
            res.append(True)
        else:
            res.append(False)
    return res


def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = f / (f_norm + 1e-9)
    return f

class VABooster(nn.Module):
    """ VABooster. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0, args=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         VABooster can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.args=args
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        # self.audio_position_embedding = audio_position_embedding
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        # self.vid_token_type_embeddings = nn.Embedding(2, hidden_dim)
        # self.txt_token_type_embeddings = nn.Embedding(1, hidden_dim)
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        self.query_embed = nn.Embedding(num_queries, 2)
        
        # torch.nn.init.xavier_uniform_(self.mu_v1)
        # if aud_dim != 0 and not self.args.audio_cat:
        #     pass

        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        if aud_dim != 0:
            self.input_audio_proj = nn.Sequential(*[
                LinearLayer(aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])

        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(args.total_prompts, hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(1, hidden_dim))

        self.affine_bottleneck = nn.Sequential(*[
                LinearLayer(hidden_dim, hidden_dim//2, layer_norm=False, dropout=0.0, relu=True),
                LinearLayer(hidden_dim//2, hidden_dim, layer_norm=False, dropout=0.0, relu=False),
            ])
        self.affine_bottleneck1 = nn.Sequential(*[
                LinearLayer(hidden_dim, hidden_dim//2, layer_norm=False, dropout=0.0, relu=True),
                LinearLayer(hidden_dim//2, hidden_dim, layer_norm=False, dropout=0.0, relu=False),
            ])
        self.affine_bottleneck2 = nn.Sequential(*[
                LinearLayer(hidden_dim, hidden_dim//2, layer_norm=False, dropout=0.0, relu=True),
                LinearLayer(hidden_dim//2, hidden_dim, layer_norm=False, dropout=0.0, relu=False),
            ])
        # self.affine_bottleneck3 = nn.Sequential(*[
        #         LinearLayer(hidden_dim, hidden_dim//2, layer_norm=False, dropout=0.0, relu=True),
        #         LinearLayer(hidden_dim//2, hidden_dim, layer_norm=False, dropout=0.0, relu=False),
        #     ])

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))
    
    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, src_aud=None, src_aud_mask=None, targets=None):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        ## For discovering real negative samples
        if vid is not None: ## for demo (run_on_video/run.py)
            _count = [v.count('_') for v in vid]
            if self.args.dset_name == 'hl':
                _position_to_cut = [find_nth(v, '_', _count[i]-1) for i, v in enumerate(vid)]
                ori_vid = [v[:_position_to_cut[i]] for i, v in enumerate(vid)]
            else:
                ori_vid = [v for v in vid]

        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)

        if src_aud is not None:
            src_aud = self.input_audio_proj(src_aud) 
            
        # v2t_sim = torch.matmul(src_vid,src_txt.permute(0,2,1)).mean(-1).unsqueeze(-1)
        # src_aud = src_aud * v2t_sim
        # src_aud = F.normalize(src_aud,dim=-1)
        # src_vid = src_vid + src_aud

        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_aud = self.position_embed(src_aud, src_aud_mask)  # (bsz, L_vid, d)

        aud2txt_similarity = torch.matmul(src_aud, src_txt.permute(0, 2, 1))
        vid2txt_similarity = torch.matmul(src_vid, src_txt.permute(0, 2, 1))

        aud2txt_ = F.softmax(torch.mean(aud2txt_similarity,dim=-1),dim=-1)
        vid2txt_ = F.softmax(torch.mean(vid2txt_similarity,dim=-1),dim=-1)

        src_aud_no_vid_query = src_aud*((1-vid2txt_).unsqueeze(-1))  #V不关A关的音频 32 75 256
        src_vid_no_aud_query = src_vid*((1-aud2txt_).unsqueeze(-1))  #V关A不关的视频 32 75 256
        src_aud_vid_query = src_aud*src_vid  #V关A关的视频音频  32 75 256

        aud_no_vid_maps = F.softmax(self.affine_bottleneck(src_aud_no_vid_query),dim=-1) #32 75 256
        vid_no_aud_maps = F.softmax(self.affine_bottleneck2(src_vid_no_aud_query),dim=-1) #32 75 256
        aud_vid_maps = F.softmax(self.affine_bottleneck1(src_aud_vid_query),dim=-1) #32 75 256

        src_vid_global = (src_vid * (vid_no_aud_maps + aud_vid_maps + 1))
        src_vid = (src_vid * (aud_no_vid_maps + aud_vid_maps + 1))

        # Input : Concat video, txt
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        
        moment_mask_ = None

        if self.args.global_cal_type == 'random_walk':
            vid_txt_sim = torch.matmul(src_vid_global,src_txt.permute(0,2,1))
            vid_txt_sim_ope = vid_txt_sim.mean(-1)
            topk_val, topkidx = torch.topk(vid_txt_sim_ope, k=1, dim=1)
            vidsrc_ = torch.zeros((len(src_vid), 1, self.hidden_dim)).cuda()
            for i in range(len(src_vid)):
                vidsrc_[i] = src_vid_global[i][topkidx[i][-1]]

        elif self.args.global_cal_type == 'cross_attention':
            ## for t_2_vid-avg sal token
            vidsrc_ = torch.zeros((len(src_vid), 1, self.hidden_dim)).cuda()
            for i in range(len(src_vid)):
                vidsrc_[i] = src_vid[i][:src_vid_mask.sum(1)[i].long()].mean(0).clone().detach()

        video_length = src_vid.shape[1]

        if targets is not None: ## train
            sentence_txt, smemory_words,moment2txt_similarity, nmoment2txt_similarity = None, None, None, None
            msrc, mpos, mmask, nmsrc, nmpos, nmmask = None, None, None, None, None, None
            hs, reference, memory, memory_global, attn_weights, memory_moment, nmmemory_moment, mmemory_frames, nmmemory_frames = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length, moment_idx=targets["relevant_clips"],
                                                                                                                  ctxtoken=vidsrc_, gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask.sum(1).long())
            
        else: ## inference
            sentence_txt, smemory_words, moment2txt_similarity, nmoment2txt_similarity = None, None, None, None
            hs, reference, memory, memory_global, attn_weights, memory_moment, nmmemory_moment, mmemory_frames, nmmemory_frames = self.transformer(src, ~mask, self.query_embed.weight, pos, video_length=video_length,
                                                                                                              ctxtoken=vidsrc_, gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask.sum(1).long())

        outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        reference_before_sigmoid = inverse_sigmoid(reference)
        tmp = self.span_embed(hs)
        outputs_coord = tmp + reference_before_sigmoid
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        if vid is not None: ## for demo (run_on_video/run.py)
            ### Neg Pairs ###
            neg_vid = ori_vid[1:] + ori_vid[:1]
            real_neg_mask = torch.Tensor(element_wise_list_equal(ori_vid, neg_vid)).to(src_txt.device)
            real_neg_mask = real_neg_mask == False
            if real_neg_mask.sum() != 0:
                src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
                src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
                src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
                mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()
                pos_neg = pos.clone()  # since it does not use actual content

                mask_neg = mask_neg[real_neg_mask]
                src_neg = src_neg[real_neg_mask]
                pos_neg = pos_neg[real_neg_mask]
                src_txt_mask_neg = src_txt_mask_neg[real_neg_mask]

                _, _, memory_neg, memory_global_neg, attn_weights_neg, _, _, _, _ = self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg, video_length=video_length,
                                                                                               ctxtoken=vidsrc_[real_neg_mask], gtoken=self.global_rep_token, gpos=self.global_rep_pos, vlen=src_vid_mask[real_neg_mask].sum(1).long())
                vid_mem_neg = memory_neg[:, :src_vid.shape[1]]
                out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
                out["src_txt_mask_neg"] = src_txt_mask_neg

                # out["t2vattnvalues_neg"] = (attn_weights_neg* (src_txt_mask_neg.unsqueeze(1).repeat(1, video_length, 1))).sum(2)
                # out["t2vattnvalues_neg"] = torch.clamp(out["t2vattnvalues_neg"], 0, 1)
            else:
                out["saliency_scores_neg"] = None
                # out["t2vattnvalues_neg"] = None
            out["real_neg_mask"] = real_neg_mask
        else:
            out["saliency_scores_neg"] = None
            # out["t2vattnvalues_neg"] = None
            out["real_neg_mask"] = None

        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["memory_moment"] = memory_moment
        out["nmmemory_moment"] = nmmemory_moment
        
        ## sentence token embeeded with text
        out["sentence_txt"] = sentence_txt

        #### FIXME
        out["mmemory_frames"] = mmemory_frames
        out["nmmemory_frames"] = nmmemory_frames
        out["smemory_words"] = smemory_words
        
        out["moment2txt_similarity"] = moment2txt_similarity
        out["nmoment2txt_similarity"] = nmoment2txt_similarity
        out["cate_attn_weights"] = attn_weights
        out["moment_mask"] = moment_mask_
        out["txt_mask"] = src_txt_mask

        out["global_rep_tokens"] = self.global_rep_token

        # if targets is not None:
        #     out["src_vid"] = mmemory_frames.permute(1, 0, 2) * moment_mask_.unsqueeze(2) + nmmemory_frames.permute(1, 0, 2) * (~(moment_mask_.unsqueeze(2).bool())).float()
        # else:
        #     out["src_vid"] = None
        out["src_vid"] = None
        
        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True, args=None):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.args=args
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        # for tvsum,
        self.use_matcher = use_matcher

        # moment sentence contrastive
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.l2_criterion = torch.nn.MSELoss().to(self.args.device)
        self.kld_criterion = torch.nn.KLDivLoss(reduction='none').to(self.args.device)
        self.bce_criterion = nn.BCELoss(reduction='none')

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        # Neg pair loss
        if outputs["saliency_scores_neg"] is not None: ## When batch size is not 1 (negative pair exists)
            vid_token_mask = outputs["video_mask"]
            real_neg_mask = outputs["real_neg_mask"]
            saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
            loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat([saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

            saliency_scores = outputs["saliency_scores"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair * 0.
            else:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
                
            
        else: ## when batch size == 1
            vid_token_mask = outputs["video_mask"]
            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)  # no drop
                pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            saliency_scores = outputs["saliency_scores"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            loss_saliency = loss_saliency + loss_rank_contrastive
            
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_moment_sentence(self, outputs, targets, indices, log=True):
        
        if outputs["smemory_words"] is not None:
            pass

        if outputs["memory_moment"] is not None:
            moment_token = outputs["memory_moment"] # b, d
            nmmemory_moment = outputs["nmmemory_moment"] # b, d
            sentence_token = outputs["sentence_txt"].squeeze(1) # b,  d

            # torch.Size([32, 256]) torch.Size([32, 256]) torch.Size([32, 256]) 
            moment_logits = F.normalize(moment_token, dim=1)
            nmoment_logits = F.normalize(nmmemory_moment, dim=1)
            sentence_logits = F.normalize(sentence_token, dim=1)

            similarity_matrix = torch.matmul(moment_logits, sentence_logits.T) # B B
            nsimilarity_matrix = torch.matmul(nmoment_logits, sentence_logits.T) # B B
            similarity_matrix = torch.cat([similarity_matrix, nsimilarity_matrix], dim=1)
            labels = torch.eye(similarity_matrix.shape[0]).to(self.args.device)
            nlabels = torch.zeros_like(nsimilarity_matrix).to(self.args.device)
            labels = torch.cat([labels, nlabels], dim=1).max(dim=1)[1]

            loss_ms_align = self.criterion(similarity_matrix, labels)

            lambda_z = 0.5
            vv_tokens_sim = torch.matmul(moment_logits, nmoment_logits.T)
            diag = torch.diag(vv_tokens_sim)
            loss_ms_align += lambda_z*torch.mean(diag)

            # global_tokens_sim = torch.matmul(global_tokens_norm, global_tokens_norm.permute(1, 0).detach())
            # for i in range(len(global_tokens_sim)):
            #     global_tokens_sim.fill_diagonal_(0)
            # loss_dummy_ortho += global_tokens_sim.abs().mean()
            # print(torch.sum(diag),torch.mean(diag))

            '''
            video_mask = outputs['video_mask']
            src_vid = outputs['src_vid']  # [bsz, L_vid, D_vid]
            moment_mask_ = torch.clamp(targets["relevant_clips"], 0, 1)

            momtokcls_pred = torch.matmul(moment_token.unsqueeze(1), src_vid.permute(0, 2, 1))  # bsz 1 L_vid
            momtokcls_label = moment_mask_
            momtokcls_logit = torch.sigmoid(momtokcls_pred)
            loss_ms_align += (self.bce_criterion(momtokcls_logit.reshape(-1), momtokcls_label.reshape(-1)) * video_mask.reshape(-1)).mean()
            '''
        else:
            loss_ms_align = 0.
        

        # v_a  =  outputs["v_a"]
        # no_v_no_a = outputs["no_v_no_a"]
        # batch_size = v_a.shape[0]
        # global_tokens_sim = torch.matmul(v_a, no_v_no_a.permute(0,2,1).detach())
        # for i in range(len(global_tokens_sim)):
        #     global_tokens_sim[i].fill_diagonal_(0)
        # # loss_ms_align = global_tokens_sim.abs().mean()
        # loss_ms_align = global_tokens_sim.norm(p=2).mean()

        loss_ms_align=0.
        global_tokens = outputs["global_rep_tokens"]
        global_tokens_norm = global_tokens / global_tokens.norm(dim=1)[:, None]
        global_tokens_sim = torch.matmul(global_tokens_norm, global_tokens_norm.permute(1, 0).detach())
        for i in range(len(global_tokens_sim)):
            global_tokens_sim.fill_diagonal_(0)
        loss_ms_align += global_tokens_sim.abs().mean()
        
        return {"loss_ms_align": loss_ms_align}

    def loss_moment2txt_sim_distill(self, outputs, targets, indices, log=True):
        
        if outputs["moment2txt_similarity"] is not None:
            
            moment2txt_similarity = outputs["moment2txt_similarity"]  # bsz L_clip 22
            moment_mask = outputs["moment_mask"].int() # bsz L_clip 1
            txt_mask = outputs["txt_mask"].unsqueeze(1).repeat(1, outputs["cate_attn_weights"].size(1), 1)  # bsz l_t

            attn_weights = outputs["cate_attn_weights"] # bsz L_clip 22
            b, L_vid, L_txt = attn_weights.size()
            loss_distill = self.kld_criterion(
                torch.log(attn_weights + 1e-6).reshape(b * L_vid, -1),
                torch.softmax(moment2txt_similarity, dim=-1).clone().detach().reshape(b * L_vid, -1)).mean(1) * moment_mask.reshape(-1)
            loss_distill = loss_distill.sum() / moment_mask.sum()

        else:
            loss_distill = 0.
        
        return {"loss_distill": loss_distill}

    def loss_orthogonal_dummy(self, outputs, targets, indices, log=True):

        loss_global_ortho=0.
        global_tokens = outputs["global_rep_tokens"]
        global_tokens_norm = global_tokens / global_tokens.norm(dim=1)[:, None]
        global_tokens_sim = torch.matmul(global_tokens_norm, global_tokens_norm.permute(1, 0).detach())
        for i in range(len(global_tokens_sim)):
            global_tokens_sim.fill_diagonal_(0)
        loss_global_ortho += global_tokens_sim.abs().mean()

        return {"loss_orthogonal_dummy": loss_global_ortho}

    # # # # # 
    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            # "contrastive_align": self.loss_contrastive_align, # can comment
            "saliency": self.loss_saliency,
            "ms_align": self.loss_contrastive_moment_sentence,
            "distill": self.loss_moment2txt_sim_distill,
            "orthogonal_dummy":self.loss_orthogonal_dummy
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)

        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency", "ms_align", "distill", "orthogonal_dummy"]
                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    if "ms_align" == loss:
                        continue
                    if "distill" == loss:
                        continue
                    if "orthogonal_dummy" == loss:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    # position_embedding, txt_position_embedding,audio_position_embedding = build_position_encoding(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    if args.a_feat_dir is None:
        model = VABooster(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            args=args
        )
    else:
        model = VABooster(
            transformer,
            position_embedding,
            txt_position_embedding,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            args=args
        )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency,
                   "loss_ms_align": args.lw_ms_align,
                   "loss_distill": args.lw_distill,
                   "loss_orthogonal_dummy":args.lw_distill}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency', 'ms_align', 'distill', 'orthogonal_dummy']
    
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For highlight detection datasets
    use_matcher = not (args.dset_name in ['youtube_uni', 'tvsum'])
        
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher, args=args
    )
    criterion.to(device)
    return model, criterion
