"""Fetch the two test split files and the pinned retrieval checkpoints on demand."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
import shutil
import tempfile
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
BASE = 'https://raw.githubusercontent.com/RYC-98/SimVLA/29a9ddc5d124ba12e2b5109f504587e103f71908/data_annotation/'


def digest(path):
    with path.open('rb') as f:
        return stream_digest(f)


def stream_digest(stream):
    h = hashlib.sha256()
    for data in iter(lambda:stream.read(8*1024*1024), b''):
        h.update(data)
    return h.hexdigest()


def fetch(path, urls, expected, size=None, split=False):
    if path.exists():
        if digest(path) == expected:
            print('Verified', path.relative_to(ROOT))
            return
        raise ValueError('Existing file differs; inspect it before replacing: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for url in urls:
        try:
            with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
                temp = Path(tmp)/'download'
                with urlopen(url, timeout=90) as response, temp.open('wb') as out:
                    shutil.copyfileobj(response, out)
                if split and path.name=='flickr30k_test.json':
                    temp.write_bytes(temp.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
                if (size is not None and temp.stat().st_size != size) or digest(temp) != expected:
                    raise ValueError('Size or SHA-256 differs from the reference')
                temp.replace(path)
            print('Installed', path.relative_to(ROOT))
            return
        except (ValueError,OSError) as error:
            failures.append(f'{url}: {error}')
    raise RuntimeError('\n'.join(failures))


def install_images(dataset, archive):
    configs=json.loads((ROOT/'configs/main.json').read_text('utf-8'))['datasets']
    spec=configs[dataset]
    annotation=ROOT/spec['annotation']
    if digest(annotation)!=spec['annotation_sha256']:
        raise ValueError('Install the verified split first')
    needed={Path(row['image']).name:row['image'] for row in json.loads(annotation.read_bytes())}
    if len(needed)!=spec['images']:
        raise ValueError('The retrieval split must have unique file names')
    image_root=ROOT/spec['image_root']

    def install(name, reader):
        member = PurePosixPath(name)
        rel=needed.get(member.name)
        if rel is None: return
        # Official archives contain the annotated relative path. Also accept
        # a wrapper directory or a flat archive of the named test images.
        expected = PurePosixPath(rel)
        if len(member.parts) > 1 and member.parts[-len(expected.parts):] != expected.parts:
            return
        del needed[member.name]
        path=image_root/rel
        if path.is_symlink() or not path.resolve().is_relative_to(image_root.resolve()):
            raise ValueError('Unsafe destination for test image: '+rel)
        if path.exists():
            with reader() as inp:
                if not path.is_file() or digest(path) != stream_digest(inp):
                    raise ValueError(f'Existing test image differs from archive member {name}: {rel}')
            return
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            temp=Path(tmp)/'image'
            with reader() as inp, temp.open('wb') as out:
                shutil.copyfileobj(inp,out)
            # Publish the complete image without replacing a concurrently created destination.
            try:
                os.link(temp, path)
            except FileExistsError:
                if path.is_symlink() or not path.is_file() or digest(path) != digest(temp):
                    raise ValueError(f'Existing test image differs from archive member {name}: {rel}')

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if not entry.is_dir():
                    install(entry.filename,lambda:z.open(entry))
    else:
        with tarfile.open(archive,'r|*') as t:
            for entry in t:
                if entry.isfile():
                    install(entry.name,lambda:t.extractfile(entry))
                if not needed: break
    if needed:
        raise ValueError(f'{len(needed)} test image names absent from the archive')
    from retrieval import check_dataset
    check_dataset(dataset)
    print('Verified ordered test images:',image_root)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('group',choices=('splits','weights','images'))
    cli.add_argument('--dataset',choices=('all','flickr30k','mscoco'),default='all')
    cli.add_argument('--archive', type=Path, help='Original ZIP or tar archive for images')
    args=cli.parse_args()
    if args.group=='images':
        if not args.archive or args.dataset=='all':
            cli.error('images requires --archive and a single --dataset')
        install_images(args.dataset,args.archive)
        return
    if args.group=='splits':
        protocol=json.loads((ROOT/'configs/main.json').read_text('utf-8'))
        for dataset,spec in protocol['datasets'].items():
            if args.dataset in ('all',dataset):
                path=ROOT/spec['annotation']
                fetch(path,[BASE+path.name],spec['annotation_sha256'],split=True)
    else:
        for rel,row in json.loads((ROOT/'configs/assets.json').read_text('utf-8')).items():
            if args.dataset=='flickr30k' and rel.endswith('_mscoco.pth') or args.dataset=='mscoco' and rel.endswith('_flickr.pth'):
                continue
            fetch(ROOT/rel,row['urls'],row['sha256'],row['size_bytes'])

if __name__=='__main__':
    main()
