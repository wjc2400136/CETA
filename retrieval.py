"""Prepare, run and read the main retrieval comparison. Metadata commands use only Python."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from vlpattack.provenance import sha256_file

PROTOCOL = json.loads((ROOT/'configs/main.json').read_text('utf-8'))
DATASETS = tuple(PROTOCOL['datasets'])
METHODS = tuple(PROTOCOL['methods'])
MODELS = tuple(PROTOCOL['models'])
EVALUATION_FILES = ('clean.npz', 'adversarial.npz', 'metrics.json')


def inputs(dataset):
    data = PROTOCOL['datasets'][dataset]
    return {**data, 'annotation':ROOT/data['annotation'], 'image_root':ROOT/data['image_root']}


def image_manifest(rows, image_root):
    records = []
    seen = set()
    root = image_root.resolve(strict=True)
    for row in rows:
        rel = PurePosixPath(row['image'])
        if rel.is_absolute() or '..' in rel.parts or rel in seen:
            raise ValueError('Unexpected image path or duplicate: '+str(rel))
        seen.add(rel)
        path = root.joinpath(rel)
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Image outside the selected tree: '+str(rel))
        records.append(f'{rel}:{path.stat().st_size}:{sha256_file(path)}')
    return hashlib.sha256('\n'.join(records).encode()).hexdigest()


def check_dataset(dataset):
    paths = inputs(dataset)
    if sha256_file(paths['annotation']) != paths['annotation_sha256']:
        raise ValueError('Annotation hash mismatch: '+str(paths['annotation']))
    rows = json.loads(paths['annotation'].read_text('utf-8'))
    if (len(rows), sum(len(r['caption']) for r in rows)) != (paths['images'],paths['captions']):
        raise ValueError('Split size or caption-group count changed')
    if image_manifest(rows, paths['image_root']) != paths['ordered_test_tree_sha256']:
        raise ValueError('Image-tree hash differs from the original test split')
    return paths


def check_assets(dataset):
    assets = json.loads((ROOT/'configs/assets.json').read_text('utf-8'))
    for rel, spec in assets.items():
        if rel.endswith('_mscoco.pth') and dataset=='flickr30k' or rel.endswith('_flickr.pth') and dataset=='mscoco':
            continue
        path = ROOT/rel
        if path.stat().st_size != spec['size_bytes'] or sha256_file(path) != spec['sha256']:
            raise ValueError('Asset hash or size mismatch: '+rel)
    bert = ROOT/'data/resources/huggingface/bert-base-uncased'
    if {p.name for p in bert.iterdir() if p.is_file()} != {Path(k).name for k in assets if k.startswith('data/resources/') }:
        raise ValueError('BERT snapshot must contain exactly the four listed files')


def reference():
    ref = ROOT/'reference/main_retrieval_r1.csv'
    if sha256_file(ref) != PROTOCOL['reference_sha256']:
        raise ValueError('The reference CSV changed')
    with ref.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    expected = set(itertools.product(MODELS,METHODS,MODELS,DATASETS,('tr','ir')))
    found = {(r['source'],r['method'],r['target'],r['dataset'],r['direction']) for r in rows}
    if found != expected or len(rows) != len(expected):
        raise ValueError('Reference row identities changed')
    count = 0
    for row in rows:
        value = float(row['asr_percent'])
        if row['k'] != '1' or not 0 <= value <= 100 or abs(value-float(row['asr_display'])) > 0.005001:
            raise ValueError('Invalid ASR or display value')
        if row['clean_hits']:
            if not math.isclose(value,100*int(row['lost_clean_hits'])/int(row['clean_hits']),abs_tol=1e-9):
                raise ValueError('ASR and saved counts disagree')
            count += 1
    print(f'{len(rows)} ASR@1 cells verified; {count} include both integer counts. {ref}')


def identity(args):
    return dict(dataset=args.dataset, method=args.method, source=args.source,
                limit=args.limit, protocol_sha256=sha256_file(ROOT/'configs/main.json'),
                source_sha256=sha256_file(ROOT/'reference/source-sha256.json'))


def verify_sources():
    for rel,digest in json.loads((ROOT/'reference/source-sha256.json').read_text('utf-8')).items():
        if sha256_file(ROOT/rel) != digest:
            raise ValueError('Source or configuration changed: '+rel)


def matches_generation(saved, current, source_field='source_sha256'):
    if saved == current:
        return True
    # Release 1a346f0 used the same generators, protocol and input assets.
    # Reconstruct its exact manifest with its original two CLI file hashes.
    previous = '40d9e26431c9f790e8463e8d12cc47aeb73fe9ce75201b8c6566b956c6250ec2'
    if source_field not in current or saved != {**current, source_field: previous}:
        return False
    manifest_bytes = (ROOT/'reference/source-sha256.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != current[source_field]:
        return False
    manifest = json.loads(manifest_bytes)
    manifest.update({
        'retrieval.py': '144efa4382d622725f9baddf2651fa13d1e971764a89acd95f079f279f96d9e0',
        'prepare_assets.py': 'eea8691c36aedb3b8f8c572fdd9724c7f2ec201165823aabac3fcfe958008dfd',
    })
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    return hashlib.sha256(encoded).hexdigest() == previous


def matrix_scope(args):
    return (DATASETS if args.dataset is None else (args.dataset,),
            METHODS if args.method is None else (args.method,),
            MODELS if args.source is None else (args.source,),
            MODELS if args.target == 'all' else (args.target,))


def matrix(args):
    datasets, methods, sources, _ = matrix_scope(args)
    for dataset, method, source in itertools.product(datasets, methods, sources):
        case = args.output/dataset/method/source
        args_case = [sys.executable, str(Path(__file__).resolve()), '--dataset',dataset,
                     '--method',method,'--source',source,'--output',str(case),'--device',args.device]
        if not (case/'generation.json').exists():
            subprocess.run(args_case+['generate'], check=True)
        subprocess.run(args_case+['evaluate','--target',args.target], check=True)


def read_ranks(path, image_count, caption_count):
    import numpy as np

    dimensions = {'text_retrieval': (image_count, caption_count),
                  'image_retrieval': (caption_count, image_count)}
    with np.load(path, allow_pickle=False) as archive:
        if sorted(archive.files) != sorted(dimensions):
            raise ValueError('Unexpected retrieval rank arrays: '+str(path))
        arrays = {}
        for name, (queries, candidates) in dimensions.items():
            ranks = archive[name]
            if (ranks.shape != (queries,) or not np.issubdtype(ranks.dtype, np.integer)
                    or (ranks < 0).any() or (ranks >= candidates).any()):
                raise ValueError(f'{path}: invalid shape, dtype or range for {name}')
            arrays[name] = ranks
    return arrays


def check_metrics(leaf, args, target, image_count, caption_count):
    rows = json.loads((leaf/'metrics.json').read_bytes())
    if (not isinstance(rows, list) or len(rows) != 2 or not all(isinstance(r, dict) for r in rows)
            or {r.get('direction') for r in rows} != {'tr', 'ir'}):
        raise ValueError('Expected one TR and one IR metric in '+str(leaf))
    clean, attacked = (
        read_ranks(leaf/name, image_count, caption_count)
        for name in ('clean.npz', 'adversarial.npz')
    )
    for row in rows:
        attr = 'text_retrieval' if row['direction'] == 'tr' else 'image_retrieval'
        hits = clean[attr] == 0
        total = int(hits.sum())
        lost = int((hits & (attacked[attr] >= 1)).sum())
        expected = dict(dataset=args.dataset, method=args.method, source=args.source, target=target,
                        k=1, clean_hits=total, lost_clean_hits=lost,
                        asr_percent=100.0 * lost / total if total else None)
        if (not expected.keys() <= row.keys() or any(row[k] != v for k, v in expected.items())
                or any(type(row[k]) is not int for k in ('k', 'clean_hits', 'lost_clean_hits'))
                or isinstance(row['asr_percent'], bool)):
            raise ValueError(f'{target} {row["direction"]}: metrics disagree with case identity or saved ranks')


def verify_evaluation(leaf, generation_sha256, args, target, image_count, caption_count):
    saved = json.loads((leaf/'complete.json').read_bytes())
    if (not isinstance(saved, dict) or saved.get('target') != target
            or saved.get('generation_sha256') != generation_sha256
            or not isinstance(saved.get('files'), dict) or set(saved['files']) != set(EVALUATION_FILES)):
        raise ValueError('Saved evaluation identity or file list mismatch: '+str(leaf))
    for name in EVALUATION_FILES:
        path = leaf/name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != saved['files'][name]:
            raise ValueError('Saved evaluation file mismatch: '+str(path))
    check_metrics(leaf, args, target, image_count, caption_count)


def run(args):
    if args.command=='plan':
        datasets, methods, sources, targets = matrix_scope(args)
        print('Datasets:', ', '.join(datasets), '\nMethods:', ', '.join(methods))
        print('Sources:', ', '.join(sources), '\nTargets:', ', '.join(targets))
        generations = len(datasets) * len(methods) * len(sources)
        evaluations = generations * len(targets)
        print('Generations:', generations)
        print('Source-target evaluations:', evaluations)
        print('TR/IR ASR@1 cells:', 2 * evaluations)
        return
    verify_sources()
    if args.command=='reference':
        reference()
        return
    if args.command=='matrix':
        matrix(args)
        return
    paths = check_dataset(args.dataset)
    check_assets(args.dataset)
    if args.command=='check-inputs':
        print(args.dataset, 'split, ordered image tree, model and MLM hashes match')
        return
    if sys.platform != 'linux':
        raise ValueError('The numerical runtime requires Linux and CUDA')
    if args.limit is not None and not 1 <= args.limit <= paths['images']:
        raise ValueError('Invalid smoke size')
    os.environ['VLPATTACK_SERVER_EXECUTION']='1'
    os.environ['VLPATTACK_ROOT']=str(ROOT)
    from vlpattack.runner.main10 import generate, evaluate
    from vlpattack.data.manifest_io import write_json_atomic
    out = args.output.resolve()
    key = identity(args)
    if args.command=='generate':
        out.mkdir(parents=True, exist_ok=False)
        generate(args, paths, out, PROTOCOL)
        rows = json.loads((out/'gallery.json').read_text('utf-8'))
        write_json_atomic(out/'generation.json', dict(identity=key, gallery_annotations=sha256_file(out/'gallery.json'),
                        gallery_images=image_manifest(rows, out/'gallery')))
        return
    receipt = json.loads((out/'generation.json').read_text('utf-8'))
    rows = json.loads((out/'gallery.json').read_text('utf-8'))
    if (not matches_generation(receipt['identity'], key) or sha256_file(out/'gallery.json') != receipt['gallery_annotations']
            or image_manifest(rows, out/'gallery') != receipt['gallery_images']):
        raise ValueError('Generation identity or saved gallery does not match')
    generation_sha = sha256_file(out/'generation.json')
    image_count = len(rows)
    caption_count = sum(len(row['caption']) for row in rows)
    for target in (MODELS if args.target=='all' else [args.target]):
        leaf = out/'evaluation'/target
        terminal = leaf/'complete.json'
        if terminal.is_file():
            verify_evaluation(leaf, generation_sha, args, target, image_count, caption_count)
            print('Verified saved target:', target, leaf/'metrics.json')
            continue
        leaf.mkdir(parents=True, exist_ok=False)
        evaluate(args, paths, out, target, leaf)
        check_metrics(leaf, args, target, image_count, caption_count)
        write_json_atomic(terminal, {'target':target, 'generation_sha256':generation_sha,
                         'files':{name:sha256_file(leaf/name) for name in EVALUATION_FILES}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','reference','check-inputs','generate','evaluate','matrix'])
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--source',choices=MODELS)
    parser.add_argument('--target',choices=(*MODELS,'all'), default='all')
    parser.add_argument('--output',type=Path,default=Path('runs/main'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit',type=int,help='Use the first N complete groups for an execution smoke')
    args=parser.parse_args()
    if args.command in ('matrix', 'plan') and args.limit is not None:
        parser.error('Matrix cases use complete splits. Use generate/evaluate for a limited smoke.')
    if args.command not in ('matrix', 'plan'):
        args.dataset = args.dataset or 'flickr30k'
        args.method = args.method or 'ceta'
        args.source = args.source or 'albef'
    run(args)

if __name__=='__main__':
    main()
