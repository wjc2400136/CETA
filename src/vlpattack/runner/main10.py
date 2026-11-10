"""Generation and saved-gallery evaluation for the ten-method retrieval comparison."""
from __future__ import annotations

import gc
from importlib import import_module
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from ..data.manifest_io import write_json_atomic
from ..data.torch_dataset import RetrievalTorchDataset
from ..modeling.loader import load_model
from ..evaluation.retrieval import evaluate_retrieval
from .common import _build_projection_matrix, _save_image, _seed_everything

CLASSES = {'pgd':'PGD', 'bert_attack':'BertAttack', 'sep_attack':'SepAttack',
           'co_attack':'CoAttack', 'sga':'SGA', 'dra':'DRA', 'sa_aet':'SA_AET',
           'simvla':'SimVLA', 'sadca':'SADCA'}


def build_baseline(method, loaded, budget):
    cls = getattr(import_module('vlpattack.attacks.' + method), CLASSES[method])
    inputs = dict(budget)
    if method == 'sadca':
        inputs['importance_batch_size'] = 32
    else:
        inputs['batch_size'] = 8
    options = {key: value for key, value in inputs.items() if key in inspect.signature(cls.__init__).parameters}
    return cls(loaded.name, loaded.model, loaded.ref_model, loaded.tokenizer, **options)


def ceta_step(loaded, images, groups, owners, caption_ids, dataset, annotation_sha):
    from ..attacks import csga as core
    from ..attacks import csga_native_inputs as native
    from ..attacks.ceta_owner_batch import DirectNativeOwnerBatchEndpoint, run_image_path_owners

    source = loaded.name
    recipe = core.resolve_recipe('BOX9_A0p50_S8_255')
    single = native.DirectNativeImageEndpoint(source, loaded.model)
    selections = []
    for i, (owner, group, ids) in enumerate(zip(owners, groups, caption_ids, strict=True)):
        expected = dataset.image2text[owner]
        contract = core.make_owner_contract(source=source, owner=owner, owner_count=len(dataset),
            caption_ids=ids, expected_caption_ids=expected,
            owner_group_manifest_sha256=annotation_sha, input_binding_sha256=annotation_sha)
        text = native.prepare_owner_text_inputs(source, loaded.model, loaded.ref_model,
            loaded.tokenizer, group, ids, expected)
        pools = tuple(core.canonical_pool_snapshot(**pool) for pool in text['pools'])
        clean = images[i:i + 1].detach()
        with torch.no_grad():
            clean_unit = single(clean)
        image_units, _ = core.collect_initial_reference(clean, clean_unit, single, contract, recipe)
        selections.append(core.prepare_full_selection(contract, recipe, pools, text['text_units'], image_units))
    endpoint = DirectNativeOwnerBatchEndpoint(source, loaded.model, native_inputs=native)
    output, _ = run_image_path_owners(images.detach(), selections, endpoint._ceta_interpolated_view_call, core=core)
    captions = [row['raw'] for selection in selections for row in core.delivery_metadata(selection)['rows']]
    return output, captions


def generate(args, paths, output, protocol):
    if args.method == 'sadca':
        import_module('vlpattack.attacks.sadca')
    _seed_everything(42)
    model = load_model(args.source, args.dataset, args.device, with_mlm=args.method != 'pgd', local_files_only=True)
    data = RetrievalTorchDataset(paths['annotation'], paths['image_root'], args.source, limit=args.limit)
    run = SimpleNamespace(seed=42, device=args.device, source_model=args.source, limit=args.limit)
    attacker = None if args.method == 'ceta' else build_baseline(args.method, model, protocol['budget'])
    if args.method == 'sadca':
        from .sadca_pool import _full_clean_pool
        core_path = inspect.getsourcefile(type(attacker))
        variant = 'v6' if args.source == 'tcl' else 'v4'
        facade = import_module('vlpattack.attacks.sadca_facade_' + variant).SADCABatch4Facade
        attacker = facade(attacker, baseline_path=core_path)
        negatives = _full_clean_pool(model, data, run, output)
    loader = DataLoader(data, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=data.collate_fn, pin_memory=torch.cuda.is_available())
    if args.method == 'sa_aet':
        projection = _build_projection_matrix(model, data, run).to(args.device)
    rows = []
    for images, groups, owners, ids, names in loader:
        images = images.to(args.device, non_blocking=True)
        flat = [text for group in groups for text in group]
        lengths = [len(group) for group in groups]
        if args.method == 'ceta':
            changed, captions = ceta_step(model, images, groups, owners, ids, data, paths['annotation_sha256'])
        elif args.method == 'sadca':
            changed, captions = attacker.attack(images, flat, lengths, source_negative_pool=negatives,
                                                image_ids=owners, text_id_groups=ids, owner_ids=owners)
        elif args.method == 'sa_aet':
            changed, captions = attacker.attack(images, flat, lengths, projection)
        else:
            changed, captions = attacker.attack(images, flat, lengths)
        if len(captions) != len(flat) or len(changed) != len(owners):
            raise ValueError('The returned image or caption count changed')
        start = 0
        for index, (owner, n, name) in enumerate(zip(owners, lengths, names, strict=True)):
            if owner != len(rows):
                raise ValueError('Unexpected generation order')
            relative = Path(name).with_suffix('.png')
            _save_image(changed[index], output / 'gallery' / relative)
            rows.append({'image':relative.as_posix(), 'caption':list(captions[start:start+n])})
            start += n
    write_json_atomic(output / 'gallery.json', rows)


def evaluate(args, paths, gallery, target, output):
    import tempfile

    _seed_everything(42)
    model = load_model(target, args.dataset, args.device, with_mlm=False, local_files_only=True)
    clean = RetrievalTorchDataset(paths['annotation'], paths['image_root'], target, limit=args.limit)
    attacked = RetrievalTorchDataset(gallery/'gallery.json', gallery/'gallery', target,
                                     limit=args.limit, clean_annotation=paths['annotation'])
    paired = []
    for label, data in [('clean', clean), ('adversarial', attacked)]:
        with tempfile.TemporaryDirectory(prefix=label+'-features-', dir=output) as temp:
            ranks = evaluate_retrieval(model, data, batch_size=8, cache_dir=Path(temp)/'features')
            ranks.validate(expected_images=len(data), expected_texts=len(data.texts))
            ranks.save(output/(label+'.npz'))
            paired.append(ranks)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    result = []
    for direction, attr in [('tr','text_retrieval'), ('ir','image_retrieval')]:
        clean_rank, adv_rank = (getattr(item, attr) for item in paired)
        hits = clean_rank < 1
        lost, total = int((hits & (adv_rank >= 1)).sum()), int(hits.sum())
        result.append(dict(dataset=args.dataset, method=args.method, source=args.source, target=target,
                           direction=direction, k=1, lost_clean_hits=lost, clean_hits=total,
                           asr_percent=100.0 * lost / total if total else None))
    write_json_atomic(output/'metrics.json', result)
