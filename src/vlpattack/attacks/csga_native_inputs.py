from __future__ import annotations


from dataclasses import dataclass


from typing import Callable


SOURCE_PROPOSAL_VERSION = 'csga-native-interpolated-view-guard-SOURCE-proposal-only-20261026-9948-v1'


INTERPOLATED_VIEW_GUARD_VERSION = 'csga-native-interpolated-view-guard-SOURCE-proposal-only-20261026-9948-v1'


INTERPOLATED_VIEW_PIXEL_TOLERANCE = 8.0 * (2.0 ** -23)


SOURCE_SPECS={
    'albef':dict(side=384,dimension=256,fields=('input_ids','attention_mask'),context=30),
    'tcl':dict(side=384,dimension=256,fields=('input_ids','attention_mask'),context=30),
    'clip_vit_b16':dict(side=224,dimension=512,fields=('input_ids',),context=77),
    'clip_rn101':dict(side=224,dimension=512,fields=('input_ids',),context=77),
}


MEAN=(0.48145466,0.4578275,0.40821073)


STD=(0.26862954,0.26130258,0.27577711)


def source_spec(source):
    if type(source) is not str or source not in SOURCE_SPECS:
        raise ValueError('One canonical retrieval source required')
    return dict(SOURCE_SPECS[source])


def validate_complete_group(originals,caption_ids,expected_caption_ids):
    """Pure metadata validation; a group may have any positive actual m_i."""
    if not isinstance(originals,(tuple,list)) or not originals:
        raise ValueError('Complete nonempty owner caption group required')
    if not isinstance(caption_ids,(tuple,list)) or not isinstance(expected_caption_ids,(tuple,list)):
        raise ValueError('Explicit complete actual and independently expected caption IDs required')
    ids=tuple(caption_ids);expected=tuple(expected_caption_ids)
    if ids!=expected or len(ids)!=len(originals) or len(set(ids))!=len(ids):
        raise ValueError('No truncated, reordered, duplicated or missing owner captions')
    if any(type(i) is not int or i<0 for i in ids):
        raise ValueError('Nonnegative canonical integer caption IDs required')
    if any(type(t) is not str or not t.strip() for t in originals):
        raise ValueError('Each original caption must contain nonblank text')
    return dict(caption_ids=ids,caption_count=len(ids))


def _count(sink,kind,**metadata):
    if sink is not None:
        if not callable(sink):raise ValueError('Optional actual operation event sink must be callable')
        sink(dict(kind=kind,**metadata))


class _CountedMLM:
    """Call-through accounting, including the existing multi-WordPiece branch."""
    def __init__(self,model,sink):self._model,self._sink=model,sink
    def __getattr__(self,name):return getattr(self._model,name)
    def __call__(self,*args,**kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids is None or ids.ndim!=2:raise ValueError('Actual two-dimensional MLM input required')
        _count(self._sink,'MLM_forward',inputs=int(ids.shape[0]),context=int(ids.shape[1]))
        return self._model(*args,**kwargs)


def complete_supplies(originals,ref_model,tokenizer,*,count=None):
    """SERVER: same complete original-position supplier, generalized m_i only."""
    import torch
    from vlpattack.attacks.bert_attack import (
        filter_characters,filter_words,get_substitues,tokenize_with_offset,
    )
    from vlpattack.data.schema import clean_caption

    if not isinstance(originals,(list,tuple)) or not originals:
        raise ValueError('One complete owner group required')
    if any(type(text) is not str or clean_caption(text,30)!=text or not text.strip() for text in originals):
        raise ValueError('Ordered original clean_caption30 strings required')
    if ref_model.training:raise ValueError('Qualified eval MLM required')
    counted=_CountedMLM(ref_model,count)
    with torch.no_grad():
        _count(count,'MLM_tokenizer',inputs=len(originals),context=30)
        inputs=tokenizer(originals,padding='max_length',truncation=True,max_length=30,
            return_tensors='pt').to(next(ref_model.parameters()).device)
        logits=counted(inputs.input_ids,attention_mask=inputs.attention_mask).logits
        scores,predictions=torch.topk(logits,10,dim=-1)
        result=[]
        for slot,text in enumerate(originals):
            words,_,keys=tokenize_with_offset(tokenizer,text,1)
            positions=[]
            for position,target in enumerate(words):
                key=keys[position]
                if target in filter_words or key[0]>28:continue
                metadata=[]
                proposals=get_substitues(predictions[slot,key[0]:key[1]],tokenizer,counted,
                    True,scores[slot,key[0]:key[1]],0.3,metadata_sink=metadata)
                accepted=[p for p in proposals if p!=target and p not in filter_words
                    and not any(c in p for c in filter_characters)]
                positions.append(dict(position=position,target=target,key=key,accepted=accepted))
            result.append(dict(slot=slot,original=text,original_words=words,
                original_wordpiece_keys=keys,position_rows=positions,
                raw_count=1+sum(len(p['accepted']) for p in positions)))
        return result


def restore_surfaces(supply):
    """Same original-first one-original-position enumeration as the old interface."""
    from vlpattack.data.schema import clean_caption
    words=supply['original_words']
    if not words or any(type(w) is not str or not w for w in words):
        raise ValueError('Exact original supplier words required')
    raw=' '.join(words)
    rows=[dict(raw=raw,native=clean_caption(raw,30),position=None,target=None,
        replacement=None,original_position_commits=0)]
    positions=set()
    for r in supply['position_rows']:
        p=r['position']
        if type(p) is not int or not 0<=p<len(words) or p in positions or r['target']!=words[p]:
            raise ValueError('Distinct original positions and exact targets required')
        positions.add(p)
        for replacement in r['accepted']:
            if type(replacement) is not str or not replacement:
                raise ValueError('Literal accepted replacement required')
            changed=words.copy();changed[p]=replacement;value=' '.join(changed)
            rows.append(dict(raw=value,native=clean_caption(value,30),position=p,
                target=words[p],replacement=replacement,original_position_commits=1))
    if len(rows)!=supply['raw_count']:raise ValueError('Complete raw enumeration mismatch')
    return rows


def _native_tokens(source,texts,tokenizer):
    """SERVER: full native text inputs, never the CLIP BERT-decode/truncate route."""
    import numpy as np
    spec=source_spec(source)
    if source.startswith('clip_'):
        from vlpattack.legacy.models.clip_model import clip
        return dict(input_ids=clip.tokenize(list(texts),77,False).numpy())
    value=tokenizer(list(texts),padding='max_length',truncation=True,max_length=30,return_tensors='np')
    return {k:np.asarray(value[k]) for k in spec['fields']}


def canonical_owner_pool(source,raw_rows,tokenizer,*,count=None):
    """Stable complete native-field dedup; keep representative raw surfaces.

    The historical CLIP overflow-only exclusion remains explicit in unencodable.
    Fusion has no such exclusions. Original overflow is a failure, not fallback.
    """
    spec=source_spec(source)
    if not raw_rows or raw_rows[0]['original_position_commits']!=0:
        raise ValueError('Original-first raw pool required')
    seen={};rows=[];representatives=[];aliases=[];overflow=[]
    for raw_index,row in enumerate(raw_rows):
        _count(count,'source_native_tokenizer',source=source,inputs=1,role='candidate')
        try:
            arrays=_native_tokens(source,[row['native']],tokenizer)
        except RuntimeError as error:
            if not source.startswith('clip_') or 'is too long for context length' not in str(error):raise
            if raw_index==0:raise ValueError('Original caption must be natively representable') from error
            overflow.append(dict(raw_index=raw_index,kind='native_context_overflow',error=str(error)))
            continue
        key=[]
        for name in spec['fields']:
            a=arrays[name]
            if a.shape!=(1,spec['context']) or a.dtype.kind not in 'iu':
                raise ValueError('Exact native integer fields/context required')
            key.append((name,tuple(int(x) for x in a[0].tolist())))
        key=tuple(key)
        if key in seen:
            aliases.append(dict(raw_index=raw_index,representative_index=seen[key]));continue
        index=len(rows);seen[key]=index;representatives.append(raw_index)
        fields=dict(key)
        saved=dict(row,index=index,raw_index=raw_index,native_token_ids=list(fields['input_ids']))
        if 'attention_mask' in fields:saved['native_attention_mask']=list(fields['attention_mask'])
        rows.append(saved)
    if not rows or rows[0]['raw_index']!=0 or rows[0]['original_position_commits']!=0:
        raise ValueError('Original canonical sentinel must remain index0')
    _count(count,'candidate_pool',source=source,raw_count=len(raw_rows),canonical_count=len(rows),
        alias_count=len(aliases),native_context_overflow_count=len(overflow))
    return dict(rows=rows,representatives=representatives,aliases=aliases,unencodable=overflow,
        raw_count=len(raw_rows),native_field_names=spec['fields'],native_context=spec['context'])


def _arrays_from_rows(source,rows):
    import numpy as np
    fields=dict(input_ids=np.asarray([r['native_token_ids'] for r in rows],dtype=np.int64))
    if not source.startswith('clip_'):
        fields['attention_mask']=np.asarray([r['native_attention_mask'] for r in rows],dtype=np.int64)
    return fields


def native_text_units(source,model,arrays,*,device,batch_size,count=None):
    """SERVER: native FP32 units with original source-specific normalization."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    spec=source_spec(source)
    if model.training or type(batch_size) is not int or batch_size<1 or tuple(arrays)!=spec['fields']:
        raise ValueError('Qualified eval source and exact ordered native fields required')
    total=len(arrays['input_ids'])
    if total<1 or any(a.shape!=(total,spec['context']) or a.dtype.kind not in 'iu' for a in arrays.values()):
        raise ValueError('Complete native integer text inputs required')
    parts=[]
    with torch.no_grad():
        for start in range(0,total,batch_size):
            batch={k:torch.as_tensor(v[start:start+batch_size],device=device).long() for k,v in arrays.items()}
            _count(count,'source_text_forward',source=source,inputs=len(batch['input_ids']),role='native_text')
            if source.startswith('clip_'):
                raw=model.encode_text(batch['input_ids'])
                unit=raw/raw.norm(dim=-1,keepdim=True) if source=='clip_rn101' else F.normalize(raw,dim=-1)
            else:
                from transformers.tokenization_utils_base import BatchEncoding
                unit=model.inference_text(BatchEncoding(batch))['text_feat']
            if unit.dtype!=torch.float32 or unit.shape!=(len(batch['input_ids']),spec['dimension']) or not bool(torch.isfinite(unit).all()):
                raise ValueError('Finite source-native FP32 text endpoints required')
            if not bool(((unit.double().norm(dim=-1)-1).abs()<=2e-6).all()):
                raise ValueError('Native unit normalization mismatch')
            parts.append(unit.detach().cpu().numpy().copy())
    return np.concatenate(parts)


def prepare_owner_text_inputs(source,model,ref_model,tokenizer,originals,caption_ids,
    expected_caption_ids,*,count=None):
    """SERVER: full m_i supply→native stable dedup→source-native candidates."""
    import numpy as np
    contract=validate_complete_group(originals,caption_ids,expected_caption_ids)
    source_spec(source)
    supplies=complete_supplies(originals,ref_model,tokenizer,count=count)
    pools=[canonical_owner_pool(source,restore_surfaces(s),tokenizer,count=count) for s in supplies]
    if len(pools)!=contract['caption_count']:raise ValueError('No missing caption pool')
    device=next(model.parameters()).device
    if source=='clip_rn101':
        # Historical RN path encoded every own-caption pool separately, batch32.
        vectors=[native_text_units(source,model,_arrays_from_rows(source,p['rows']),
            device=device,batch_size=32,count=count) for p in pools]
    else:
        sizes=[len(p['rows']) for p in pools]
        joined=native_text_units(source,model,_arrays_from_rows(source,[r for p in pools for r in p['rows']]),
            device=device,batch_size=160,count=count)
        vectors=list(np.split(joined,np.cumsum(sizes)[:-1]))
    for p,v in zip(pools,vectors,strict=True):
        if len(v)!=len(p['rows']):raise ValueError('Native vectors do not cover canonical rows')
    return dict(source=source,caption_ids=contract['caption_ids'],originals=tuple(originals),
        supplies=supplies,pools=pools,text_units=vectors)


def _native_pixel_guard_observation(pixels,finite_mask,*,tolerance,interpolated_view):
    """Actual scalar diagnostics only; never repair or replace endpoint pixels."""
    observed=pixels.detach()
    finite_all=bool(finite_mask.all())
    minimum=float(observed.min());maximum=float(observed.max())
    if finite_all:
        finite_minimum,finite_maximum=minimum,maximum
    else:
        finite_values=observed[finite_mask]
        finite_minimum=float(finite_values.min()) if finite_values.numel() else None
        finite_maximum=float(finite_values.max()) if finite_values.numel() else None
    return dict(guard_source_version=SOURCE_PROPOSAL_VERSION,
        interpolated_view=interpolated_view,tolerance=tolerance,finite_all=finite_all,
        min=minimum,max=maximum,finite_min=finite_minimum,finite_max=finite_maximum,
        nonfinite_count=int((~finite_mask).sum()),
        below_zero_count=int((observed<0.0).sum()),above_one_count=int((observed>1.0).sum()),
        below_allowed_count=int((observed < -tolerance).sum()),
        above_allowed_count=int((observed > 1.0+tolerance).sum()))


@dataclass
class DirectNativeImageEndpoint:
    source:str
    model:object
    count:Callable|None=None

    def __post_init__(self):
        source_spec(self.source)
        if self.model.training:raise ValueError('Qualified eval model required')
        if self.count is not None and not callable(self.count):raise ValueError('Optional callable event sink required')

    _ceta_interpolated_view_guard_version = INTERPOLATED_VIEW_GUARD_VERSION

    def _ceta_interpolated_view_call(self,pixels):
        """Explicit one-argument native image-path endpoint; clean/BOX stay strict."""
        return self(pixels,interpolated_view=True)

    def __call__(self,pixels,*,interpolated_view=False):
        """SERVER: native carrier→normalization→complete native endpoint."""
        if type(interpolated_view) is not bool:
            raise ValueError('Explicit boolean interpolated-view guard mode required')
        import torch
        spec=source_spec(self.source)
        if not isinstance(pixels,torch.Tensor) or pixels.dtype!=torch.float32 or tuple(pixels.shape)!=(1,3,spec['side'],spec['side']):
            raise ValueError('Actual source-native RGB carrier required; no implicit224-to384 resize')
        tolerance=INTERPOLATED_VIEW_PIXEL_TOLERANCE if interpolated_view else 0.0
        finite_mask=torch.isfinite(pixels)
        if not bool(finite_mask.all()) or not bool(((pixels >= -tolerance)&(pixels <= 1.0+tolerance)).all()):
            observation=_native_pixel_guard_observation(pixels,finite_mask,
                tolerance=tolerance,interpolated_view=interpolated_view)
            # repr keeps actual nan/inf scalars inside the exception STRING,
            # which existing strict-JSON error receipts can preserve safely.
            raise ValueError('Finite valid native carrier pixels required; actual_guard_observation='+repr(observation))
        event_metadata={}
        if interpolated_view and bool(((pixels<0.0)|(pixels>1.0)).any()):
            event_metadata['accepted_interpolation_roundoff']=_native_pixel_guard_observation(
                pixels,finite_mask,tolerance=tolerance,interpolated_view=True)
        mean=pixels.new_tensor(MEAN).view(1,3,1,1);std=pixels.new_tensor(STD).view(1,3,1,1)
        _count(self.count,'source_image_forward',source=self.source,inputs=1,carrier_side=spec['side'],**event_metadata)
        unit=self.model.inference_image((pixels-mean)/std)['image_feat']
        if unit.dtype!=torch.float32 or tuple(unit.shape)!=(1,spec['dimension']) or not bool(torch.isfinite(unit).all()):
            raise ValueError('Finite complete source-native FP32 image endpoint required')
        if not bool(((unit.detach().double().norm(dim=-1)-1).abs()<=2e-6).all()):
            raise ValueError('Native image unit normalization mismatch')
        if pixels.requires_grad and not unit.requires_grad:
            raise ValueError('Native endpoint detached the original pixel graph')
        return unit
