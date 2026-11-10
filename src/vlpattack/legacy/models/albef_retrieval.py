from functools import partial
from vlpattack.legacy.models.vit import VisionTransformer
from vlpattack.legacy.models.xbert import BertConfig, BertModel

import torch
import torch.nn.functional as F
from torch import nn


class ALBEF(nn.Module):
    def __init__(self, config=None, text_encoder=None, tokenizer=None):
        super().__init__()
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        
        self.tokenizer = tokenizer
        
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel(config=bert_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)

    def inference_text(self, text_input):
        text_output = self.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_embed = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embed[:, 0, :]), dim=-1)
        return {'text_embed': text_embed, 'text_feat': text_feat}

    def inference_image(self, image):
        image_embed = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embed[:, 0, :]), dim=-1)
        return {'image_embed': image_embed, 'image_feat': image_feat}

    def inference(self, image, text_input=None, use_embeds=False):
        if not use_embeds:
            text_output = self.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        else:
            text_output = self.text_encoder(inputs_embeds=text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        
        text_embed = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embed[:, 0, :]), dim=-1)

        image_embed = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embed[:, 0, :]), dim=-1)

        encoder_att = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(image.device)

        fusion_output = self.text_encoder(
            encoder_embeds=text_embed,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=image_embed,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode='fusion'
        )

        return {'text_embed': text_embed, 'image_embed': image_embed, 
                'text_feat': text_feat, 'image_feat': image_feat, 
                'fusion_output': fusion_output.last_hidden_state}
