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


def matrix(args):
    for dataset, method, source in itertools.product(DATASETS, METHODS, MODELS):
        case = args.output/dataset/method/source
        args_case = [sys.executable, str(Path(__file__).resolve()), '--dataset',dataset,
                     '--method',method,'--source',source,'--output',str(case),'--device',args.device]
        if not (case/'generation.json').exists():
            subprocess.run(args_case+['generate'], check=True)
        subprocess.run(args_case+['evaluate','--target','all'], check=True)


def run(args):
    if args.command=='plan':
        print('Datasets:', ', '.join(DATASETS),'\nMethods:', ', '.join(METHODS))
        print('Models as sources and targets:', ', '.join(MODELS))
        print('Full matrix: 80 generations, 320 source-target evaluations, 640 TR/IR ASR@1 cells.')
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
    if (receipt['identity'] != key or sha256_file(out/'gallery.json') != receipt['gallery_annotations']
            or image_manifest(rows, out/'gallery') != receipt['gallery_images']):
        raise ValueError('Generation identity or saved gallery does not match')
    for target in (MODELS if args.target=='all' else [args.target]):
        leaf = out/'evaluation'/target
        terminal = leaf/'complete.json'
        if terminal.is_file():
            saved = json.loads(terminal.read_text())
            if saved['generation_sha256'] != sha256_file(out/'generation.json') or any(
                    sha256_file(leaf/name)!=digest for name,digest in saved['files'].items()):
                raise ValueError('Saved evaluation identity mismatch: '+str(leaf))
            print('Verified saved target:', target, leaf/'metrics.json')
            continue
        leaf.mkdir(parents=True, exist_ok=False)
        evaluate(args, paths, out, target, leaf)
        write_json_atomic(terminal, {'target':target, 'generation_sha256':sha256_file(out/'generation.json'),
                         'files':{name:sha256_file(leaf/name) for name in ('clean.npz','adversarial.npz','metrics.json')}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','reference','check-inputs','generate','evaluate','matrix'])
    parser.add_argument('--dataset',choices=DATASETS, default='flickr30k')
    parser.add_argument('--method',choices=METHODS, default='ceta')
    parser.add_argument('--source',choices=MODELS, default='albef')
    parser.add_argument('--target',choices=(*MODELS,'all'), default='all')
    parser.add_argument('--output',type=Path,default=Path('runs/main'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit',type=int,help='Use the first N complete groups for an execution smoke')
    args=parser.parse_args()
    if args.command=='matrix' and args.limit is not None:
        parser.error('The full matrix uses complete splits')
    run(args)

if __name__=='__main__':
    main()
