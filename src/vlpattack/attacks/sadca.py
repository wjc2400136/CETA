"""Official SADCA port plus the named benchmark compatibility wrapper.

Source: LiYuanBoJNU/SADCA, commit a3072c64f6355753742609d637d43f2837d7b6a1.
Upstream MIT LICENSE is retained in the isolated official source preparation.
Candidate variant: release_mlm_offset1_uniform30_suite_budget_serial1_v2.
Versioned suite epsilon/step and native text-budget parameters; ordered
single-image native calls and final original-word-slot witness only.
Previously disclosed changes: all-source BERT/MLM input
max_length30 (native CLIP encoding stays77), MLM span offset1, source-device
kernel allocation, silent progress output.
Native alignment, dynamic schedule, loss reductions, augmentation, initialization
and candidate enumeration are retained below. See runtime port-fidelity.json.
"""
from __future__ import annotations

import sys

from ..sadca_word_witness import source_word_witness

import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F
import random
import time
import scipy.stats as st
import os
from collections import defaultdict

class Attacker():
    def __init__(self, model, img_attacker, txt_attacker, tokenizer, lam, dynamic_num):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.lam = lam
        self.dynamic_num = dynamic_num
        self.tokenizer = tokenizer

    # def save_img(self, img_name, norm_img):
    #     pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
    #     pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
    #     img_path = "./flickr30k-adv/"
    #     pil_img.save(img_path + img_name)

    def attack(self, imgs, txts, txt2img, neg_img_embeddings, neg_txt_embeddings, device='cpu', max_length=30, masks=None, **kwargs):
        start_time = time.time()
        with torch.no_grad():

            txts_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']

            # origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            # img_supervisions = origin_img_output['image_feat'][txt2img]


        ali_imgs = self.img_attacker.img_pos(self.model, imgs, txt2img, device, txt_embeds=txt_supervisions)
        with torch.no_grad():
            ali_img_output = self.model.inference_image(self.img_attacker.normalization(ali_imgs))
            ali_img_supervisions = ali_img_output['image_feat'][txt2img]

        adv_imgs = imgs.clone()
        momentum = 0
        for i in range(self.dynamic_num):
            # Author progress print removed; no numerical or RNG change.
            with torch.no_grad():
                adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
                adv_img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=ali_img_supervisions,
                                                            adv_img_embeds=adv_img_supervisions,
                                                            txt_embeds=txt_supervisions,
                                                            neg_img_embeds=neg_img_embeddings)
            # Author progress print removed; no numerical or RNG change.

            with torch.no_grad():
                adv_txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                             max_length=max_length, return_tensors="pt").to(device)
                adv_txts_output = self.model.inference_text(adv_txts_input)
                adv_txt_supervisions = adv_txts_output['text_feat']
            adv_imgs, momentum = self.img_attacker.txt_guided_attack(self.model, adv_imgs, txt2img, device, txt_embeds=txt_supervisions,
                                                           adv_txt_embeds=adv_txt_supervisions,
                                                           img_embeds=ali_img_supervisions,
                                                           neg_txt_embeds=neg_txt_embeddings,
                                                           clean_imgs=imgs,
                                                           ali_imgs=ali_imgs, momentum=momentum,
                                                           txts=txts, adv_txts=adv_txts, tokenizer=self.tokenizer)


        end_time = time.time()
        execuate_time = end_time - start_time
        # Author progress print removed; no numerical or RNG change.

        # print(adv_imgs.max(), adv_imgs.min())
        # print(txts)
        # print(adv_txts)
        # self.save_img('0.png',adv_imgs[0])
        # sys.exit()
        return adv_imgs, adv_txts, execuate_time




class ImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255, sample_numbers=5, lam=0.2, input_trans_num=10,  max_length=30, device=None):
        self.normalization = normalization
        self.device = torch.device("cpu" if device is None else device)
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.sample_numbers = sample_numbers
        self.lam = lam
        self.max_length = max_length
        self.input_trans_num = input_trans_num

        # transformation
        self.num_block = 3
        self.kernel = self.gkern()
        self.op = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180,
                   self.scale, self.add_noise, self.BSR]

    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2, 3))

    def scale(self, x):
        return torch.rand(1)[0] * x

    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        # return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def blur(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding='same', groups=3)


    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)
    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips
    def BSR(self, x):
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat(
            [torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips],
            dim=dims[0])

    def local_semantic_augmentation(self, x):
        B, C, H, W = x.shape
        local_img_transform = transforms.RandomResizedCrop(H, scale=(0.4, 0.8))

        x = local_img_transform(x)
        chosen = np.random.randint(0, high=len(self.op), dtype=np.int32)
        # x = self.op[chosen](x)
        augmentation_img = self.op[chosen](x)

        return augmentation_img

    def mix_texts_in_groups(self, texts, txt2img, method='concat', alpha=0.7):
        mixed_texts = []
        # Group text by image ID
        img2texts = defaultdict(list)
        for txt, img_id in zip(texts, txt2img):
            img2texts[img_id].append(txt)

        # The text inside each set of images is blended
        for group in img2texts.values():
            shuffled_group = group.copy()
            random.shuffle(shuffled_group)
            for original, shuffled in zip(group, shuffled_group):
                if method == 'concat':
                    mixed = original + ' ' + shuffled
                elif method == 'interpolate':
                    l = int(len(original) * alpha)
                    r = int(len(shuffled) * (1 - alpha))
                    mixed = original[:l] + shuffled[-r:]
                else:
                    raise ValueError("Unsupported method")
                mixed_texts.append(mixed)
        return mixed_texts

    def loss_func_contrast(self, adv_imgs_embeds, txts_embeds, txt2img, neg_txts_embeds):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()

        neg_sim_matrix = adv_imgs_embeds @ neg_txts_embeds.T
        it_labels = torch.zeros(neg_sim_matrix.shape).to(device)
        k = int(len(neg_txts_embeds) / len(adv_imgs_embeds))
        txt2img_neg = torch.arange(len(adv_imgs_embeds)).repeat_interleave(k)
        for i in range(len(txt2img_neg)):
            it_labels[txt2img_neg[i], i] = 1
        loss_IaTcneg = (neg_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos + self.lam * loss_IaTcneg

        return loss

    def loss_func_self(self, adv_imgs_embeds, imgs_embeds, txt2img):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ imgs_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()

        loss = 0.2 * loss_IaTcpos
        # print(loss)
        return loss

    def loss_func_old(self, adv_imgs_embeds, txts_embeds, txt2img):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        # print(it_labels)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        # print(it_labels)
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        # print(loss_IaTcpos)
        # sys.exit()
        return loss

    def txt_guided_attack(self, model, imgs, txt2img, device, txt_embeds=None, adv_txt_embeds=None, img_embeds=None, neg_txt_embeds=None,
                          clean_imgs=None, ali_imgs=None, momentum=None,
                          txts=None, adv_txts=None, tokenizer=None):

        model.eval()

        b, _, h, w = imgs.shape

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(
            device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        input_trans_num = self.input_trans_num
        for step in range(self.steps):  # self.steps=10
            adv_imgs.requires_grad_()
            scaled_imgs = torch.cat([adv_imgs, ] + [self.local_semantic_augmentation(adv_imgs) for _ in range(input_trans_num - 1)], 0)

            with torch.no_grad():
                mix_txts = self.mix_texts_in_groups(txts, txt2img)
                mix_adv_txts = self.mix_texts_in_groups(adv_txts, txt2img)
                mix_txts_input = tokenizer(mix_txts, padding='max_length', truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(device)
                mix_txts_output = model.inference_text(mix_txts_input)
                mix_txt_supervisions = mix_txts_output['text_feat']
                txt_embeds = mix_txt_supervisions

                mix_adv_txts_input = tokenizer(mix_adv_txts, padding='max_length', truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(device)
                mix_adv_txts_output = model.inference_text(mix_adv_txts_input)
                mix_adv_txt_supervisions = mix_adv_txts_output['text_feat']
                adv_txt_embeds = mix_adv_txt_supervisions

            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))

            else:
                adv_imgs_output = model.inference_image(scaled_imgs)

            adv_imgs_embeds = adv_imgs_output['image_feat']

            model.zero_grad()
            with torch.enable_grad():
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(int(len(scaled_imgs) / b)):
                    loss_item = self.loss_func_contrast(adv_imgs_embeds[i * b:i * b + b], adv_txt_embeds, txt2img, neg_txt_embeds) + \
                                self.loss_func_contrast(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img, neg_txt_embeds)
                    loss += loss_item
            loss.backward()
            # Author progress print removed; no numerical or RNG change.
            grad = adv_imgs.grad

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = momentum * 1.0 + grad
            perturbation = self.step_size * momentum.sign()
            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, clean_imgs - self.eps), clean_imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        # Author progress print removed; no numerical or RNG change.
        return adv_imgs.detach(), momentum

    def img_pos(self, model, imgs, txt2img, device, txt_embeds=None):
        model.eval()
        b, _, _, _ = imgs.shape

        col_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        col_imgs = torch.clamp(col_imgs, 0.0, 1.0)
        momentum = 0
        # for step in range(self.steps):  # self.steps=10
        for step in range(5):  # self.steps=10

            col_imgs.requires_grad_()

            if self.normalization is not None:
                col_imgs_output = model.inference_image(self.normalization(col_imgs))
            else:
                col_imgs_output = model.inference_image(col_imgs)

            col_imgs_embeds = col_imgs_output['image_feat']
            model.zero_grad()
            with torch.enable_grad():
                loss = - self.loss_func_old(col_imgs_embeds, txt_embeds, txt2img)
            loss.backward()
            # print("loss", loss)
            grad = col_imgs.grad

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = momentum * 1.0 + grad
            perturbation = self.step_size * momentum.sign()

            # perturbation = self.step_size * grad.sign()
            col_imgs = col_imgs.detach() + perturbation
            col_imgs = torch.min(torch.max(col_imgs, imgs - self.eps), imgs + self.eps)
            col_imgs = torch.clamp(col_imgs, 0.0, 1.0)
        # Author progress print removed; no numerical or RNG change.

        # print(f"The function execution time: {elapsed_time} seconds")
        return col_imgs.detach()

    def save_img(self, img_name, norm_img):
        pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
        img_path = "./mscoco_imgs/"
        pil_img.save(img_path + img_name)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2], lam=0.2):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios
        self.lam = lam

    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None, txt_embeds=None, neg_img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        final_source_word_witnesses = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                loss = self.loss_func(replace_embeds, adv_img_embeds, i, neg_img_embeds) + self.loss_func(replace_embeds, img_embeds, i, neg_img_embeds)

                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))
            # Observe the native final list slots without changing candidate
            # selection, counting, RNG or the subsequent image update.
            final_source_word_witnesses.append(source_word_witness(
                text, words, final_words, change
            ))
        self.last_source_word_witnesses = final_source_word_witnesses
        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label, neg_img_embeds):

        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss_TaIcneg = 0
        for k in range(neg_img_embeds.shape[1]):
            loss_neg = txt_embeds.mul(neg_img_embeds[label][k].repeat(len(txt_embeds), 1)).sum(-1)
            loss_TaIcneg += loss_neg

        loss = loss_TaIcpos + self.lam * (loss_TaIcneg / neg_img_embeds.shape[1])

        return loss


    def loss_txt_func(self, txt_embeds, txt_embeds2):
        loss_TaIcpos = -txt_embeds.mul(txt_embeds2).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))

                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 1  # Explicit benchmark correction: skip the prepended CLS logit.
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words

# Benchmark adapter. The native classes/functions above remain the authority for
# alignment, dynamic interactions, candidate search, augmentation and updates.
from collections.abc import Mapping
import math
import operator


CANONICAL_VARIANT = "release_mlm_offset1_uniform30_suite_budget_serial1_v2"


def _pool_field(pool, name, alternate=None):
    if isinstance(pool, Mapping):
        if name in pool:
            return pool[name]
        if alternate is not None and alternate in pool:
            return pool[alternate]
    else:
        if hasattr(pool, name):
            return getattr(pool, name)
        if alternate is not None and hasattr(pool, alternate):
            return getattr(pool, alternate)
    raise ValueError("SADCA clean source pool is missing " + name)


def _integer_id(value):
    if isinstance(value, bool):
        raise ValueError("Boolean values are not source-pool ids")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("Source-pool ids must be integers") from error


class SADCA:
    """Pinned author SADCA with the explicitly named benchmark adaptations.

    This wrapper only owns input validation, the author evaluation script's
    clean-current-batch negative sampling prelude, and the two-value runner API.
    Native image/text algorithms remain in Attacker/ImageAttacker/TextAttacker.
    """

    implementation_variant = CANONICAL_VARIANT

    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        epsilon=2 / 255,
        steps=10,
        step_size=0.5 / 255,
        norm_type="linfty",
        random_init=True,
        max_length=30,
        threshold=0.3,
        num_perturb=1,
        topk=10,
        importance_batch_size=32,
        dynamic_num=5,
        topk_neg=20,
        lam=0.2,
        input_trans_num=10,
    ):
        fixed_integer_values = {
            "steps": (steps, 10),
            "max_length": (max_length, 30),
            "topk": (topk, 10),
            "importance_batch_size": (importance_batch_size, 32),
            "dynamic_num": (dynamic_num, 5),
            "topk_neg": (topk_neg, 20),
            "input_trans_num": (input_trans_num, 10),
        }
        for name, (actual, expected) in fixed_integer_values.items():
            if isinstance(actual, bool) or actual != expected:
                raise ValueError(CANONICAL_VARIANT + " requires " + name + "=" + str(expected))
        if isinstance(num_perturb, bool) or not isinstance(num_perturb, int) or num_perturb not in range(1, 6):
            raise ValueError("SADCA suite text budget must be an integer from 1 to 5")
        if isinstance(epsilon, bool) or isinstance(step_size, bool):
            raise ValueError("SADCA pixel budget and step cannot be booleans")
        epsilon, step_size = float(epsilon), float(step_size)
        if not any(math.isclose(epsilon, pixel / 255, rel_tol=0.0, abs_tol=1e-15)
                   for pixel in (1, 2, 4, 8, 16)):
            raise ValueError("SADCA epsilon is outside the selected existing suite grid")
        if not math.isclose(step_size, epsilon / 4, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("SADCA existing suite step must equal epsilon / 4")
        for name, actual, expected in (
            ("threshold", threshold, 0.3),
            ("lam", lam, 0.2),
        ):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(CANONICAL_VARIANT + " has a fixed " + name)
        if norm_type != "linfty" or random_init is not True:
            raise ValueError("Canonical SADCA requires linfty and its native random starts")
        if ref_model is None:
            raise ValueError("SADCA requires the frozen BERT masked-language model")
        supported = {"albef", "tcl", "clip_vit_b16", "clip_rn101", "clip_vit", "clip_cnn"}
        if str(model_name).lower() not in supported:
            raise ValueError("Canonical SADCA supports its four declared source families")

        self.model_name = model_name
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.epsilon = float(epsilon)
        self.steps = steps
        self.step_size = float(step_size)
        self.max_length = max_length
        self.topk_neg = topk_neg
        self.num_perturb = num_perturb
        self.native_attack_images_per_call = 1
        self.normalization = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.image_attacker = ImageAttacker(
            self.normalization,
            eps=self.epsilon,
            steps=steps,
            step_size=self.step_size,
            lam=lam,
            input_trans_num=input_trans_num,
            max_length=max_length,
            device=self.device,
        )
        self.text_attacker = TextAttacker(
            ref_model,
            tokenizer,
            cls=False,
            max_length=max_length,
            number_perturbation=num_perturb,
            topk=topk,
            threshold_pred_score=threshold,
            batch_size=importance_batch_size,
            lam=lam,
        )
        self.native_attacker = Attacker(
            model, self.image_attacker, self.text_attacker, tokenizer, lam, dynamic_num
        )
        self._pool_object = None
        self._pool_view = None
        self.last_native_attack_seconds = None
        self.last_negative_selection = None
        self.last_final_source_word_witnesses = None

    def _validated_pool(self, pool):
        image_features = _pool_field(pool, "image_features")
        text_features = _pool_field(pool, "text_features")
        if self._pool_object is pool:
            if (
                image_features is not self._pool_view["image_features"]
                or text_features is not self._pool_view["text_features"]
            ):
                raise ValueError("An immutable SADCA pool replaced its feature tensors")
            return self._pool_view
        for label, features in (("image", image_features), ("text", text_features)):
            if not isinstance(features, torch.Tensor) or features.ndim != 2:
                raise ValueError("SADCA pool " + label + " features must be a rank-2 tensor")
            if features.dtype != torch.float32 or features.device != self.device:
                raise ValueError("SADCA pool features must be source-device float32")
            if features.requires_grad:
                raise ValueError("SADCA clean pool features must be detached")
        if image_features.shape[1] != text_features.shape[1]:
            raise ValueError("SADCA clean source image/text feature dimensions differ")
        image_ids = tuple(
            _integer_id(value)
            for value in _pool_field(pool, "image_ids", "ordered_image_ids")
        )
        caption_ids = tuple(
            _integer_id(value)
            for value in _pool_field(pool, "caption_ids", "ordered_caption_ids")
        )
        if (
            len(image_ids) != len(image_features)
            or len(caption_ids) != len(text_features)
            or len(set(image_ids)) != len(image_ids)
            or len(set(caption_ids)) != len(caption_ids)
        ):
            raise ValueError("SADCA pool ordered ids do not match its feature rows")
        if len(image_ids) - 1 < self.topk_neg:
            raise ValueError("SADCA full clean pool has fewer than K non-self images")
        image_rows = {value: row for row, value in enumerate(image_ids)}
        caption_rows = {value: row for row, value in enumerate(caption_ids)}
        raw_caption_to_image = _pool_field(pool, "caption_to_image")
        raw_image_to_captions = _pool_field(pool, "image_to_captions")
        if isinstance(raw_caption_to_image, Mapping):
            if set(raw_caption_to_image) != set(caption_ids):
                raise ValueError("SADCA caption_to_image does not cover the ordered caption ids")
            caption_to_image = {
                value: _integer_id(raw_caption_to_image[value]) for value in caption_ids
            }
        else:
            values = tuple(raw_caption_to_image)
            if len(values) != len(caption_ids):
                raise ValueError("SADCA caption_to_image length differs")
            caption_to_image = {
                value: _integer_id(owner)
                for value, owner in zip(caption_ids, values, strict=True)
            }
        if isinstance(raw_image_to_captions, Mapping):
            if set(raw_image_to_captions) != set(image_ids):
                raise ValueError("SADCA image_to_captions does not cover the ordered image ids")
            image_to_captions = {
                value: tuple(_integer_id(x) for x in raw_image_to_captions[value])
                for value in image_ids
            }
        else:
            values = tuple(raw_image_to_captions)
            if len(values) != len(image_ids):
                raise ValueError("SADCA image_to_captions length differs")
            image_to_captions = {
                value: tuple(_integer_id(x) for x in group)
                for value, group in zip(image_ids, values, strict=True)
            }
        seen = []
        for image_id, group in image_to_captions.items():
            if not group or len(caption_ids) - len(group) < self.topk_neg:
                raise ValueError("SADCA needs a nonempty caption group and K non-self captions")
            for caption_id in group:
                if caption_id not in caption_rows or caption_to_image[caption_id] != image_id:
                    raise ValueError("SADCA source pool image/caption maps disagree")
            seen.extend(group)
        if len(seen) != len(caption_ids) or set(seen) != set(caption_ids):
            raise ValueError("SADCA source pool maps repeat or omit caption ids")
        view = {
            "image_features": image_features,
            "text_features": text_features,
            "image_rows": image_rows,
            "caption_rows": caption_rows,
            "caption_to_image": caption_to_image,
            "image_to_captions": image_to_captions,
            "identity": _pool_field(pool, "identity"),
        }
        self._pool_object = pool
        self._pool_view = view
        return view

    def _sample_negative_embeddings(self, images, texts, txt2img, image_ids, text_id_groups, pool):
        # This intentionally retains the author eval_SADCA.py:159-212 prelude:
        # two current-batch forwards and both normalized similarity matrices are
        # executed even though the active negative selection is uniform random.
        with torch.no_grad():
            txts_input = self.tokenizer(
                texts, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output["text_feat"]
            origin_img_output = self.model.inference_image(self.normalization(images))
            img_supervisions = origin_img_output["image_feat"]

        all_img_supervisions = pool["image_features"]
        all_txt_supervisions = pool["text_features"]
        txt2img_global = [
            pool["image_rows"][image_ids[local_image]]
            for local_image in txt2img
        ]
        text_ids_groups = [
            [pool["caption_rows"][value] for value in group]
            for group in text_id_groups
        ]
        topk = self.topk_neg
        sim_txt2img = F.normalize(txt_supervisions, dim=1) @ F.normalize(all_img_supervisions, dim=1).T
        txt2img_tensor = torch.tensor(txt2img_global)
        mask = torch.zeros_like(sim_txt2img, dtype=torch.bool)
        mask[torch.arange(len(txt2img)), txt2img_tensor] = True
        sim_txt2img[mask] = -float("inf")
        neg_img_idx = []
        for i in range(sim_txt2img.size(0)):
            available_idx = (~mask[i]).nonzero(as_tuple=True)[0]
            # The randperm is deliberately created on CPU, as in the release.
            rand_idx = available_idx[torch.randperm(len(available_idx))[:topk]]
            neg_img_idx.append(rand_idx)
        neg_img_idx = torch.stack(neg_img_idx)

        sim_img2txt = F.normalize(img_supervisions, dim=1) @ F.normalize(all_txt_supervisions, dim=1).T
        img2txt_mask = torch.zeros_like(sim_img2txt, dtype=torch.bool)
        for i, text_ids in enumerate(text_ids_groups):
            img2txt_mask[i, text_ids] = True
        sim_img2txt[img2txt_mask] = -float("inf")
        neg_txt_idx = []
        for i in range(sim_img2txt.size(0)):
            available_idx = (~img2txt_mask[i]).nonzero(as_tuple=True)[0]
            rand_idx = available_idx[torch.randperm(len(available_idx))[:topk]]
            neg_txt_idx.append(rand_idx)
        neg_txt_idx = torch.stack(neg_txt_idx).view(-1)

        neg_img_embeddings = all_img_supervisions[neg_img_idx]
        neg_txt_embeddings = all_txt_supervisions[neg_txt_idx]
        # Observation only, after both native gathers. These are pool row indices,
        # not global ids. CUDA-to-host synchronization is charged by the runner.
        self.last_negative_selection = {
            "negative_image_rows_by_caption": neg_img_idx.detach().cpu().tolist(),
            "negative_caption_rows": neg_txt_idx.detach().cpu().tolist(),
        }
        return neg_img_embeddings, neg_txt_embeddings

    def attack(
        self,
        images,
        texts,
        text_group_lens,
        *,
        source_negative_pool,
        image_ids,
        text_id_groups,
    ):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or len(images) != 1:
            raise ValueError("SADCA native generation unit is one complete caption group")
        if images.dtype != torch.float32:
            raise ValueError("SADCA source pixels must be float32")
        lengths = tuple(_integer_id(value) for value in text_group_lens)
        image_ids = tuple(_integer_id(value) for value in image_ids)
        text_id_groups = tuple(
            tuple(_integer_id(value) for value in group) for group in text_id_groups
        )
        texts = list(texts)
        if (
            len(lengths) != 1 or lengths[0] < 1 or len(texts) != lengths[0]
            or len(image_ids) != 1 or len(text_id_groups) != 1
            or len(text_id_groups[0]) != lengths[0]
            or any(not isinstance(text, str) for text in texts)
        ):
            raise ValueError("SADCA input caption groups and ids disagree")
        pool = self._validated_pool(source_negative_pool)
        if image_ids[0] not in pool["image_rows"]:
            raise ValueError("SADCA batch image is missing from the full clean pool")
        if text_id_groups[0] != pool["image_to_captions"][image_ids[0]]:
            raise ValueError("SADCA batch must preserve all clean captions in their frozen order")
        images = images.detach().to(self.device)
        self.model.eval()
        self.ref_model.eval()
        txt2img = [local for local, count in enumerate(lengths) for _ in range(count)]
        neg_images, neg_texts = self._sample_negative_embeddings(
            images, texts, txt2img, image_ids, text_id_groups, pool
        )
        adv_images, adv_texts, native_seconds = self.native_attacker.attack(
            images, texts, txt2img, neg_images, neg_texts,
            device=self.device, max_length=self.max_length,
        )
        self.last_native_attack_seconds = native_seconds
        # TextAttacker overwrites this at every native text call. After the
        # full dynamic attack, this is the final (fifth) round only.
        self.last_final_source_word_witnesses = copy.deepcopy(
            self.text_attacker.last_source_word_witnesses
        )
        return adv_images, adv_texts

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)
