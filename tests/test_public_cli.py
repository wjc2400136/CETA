"""Offline entry-point regressions. Requires NumPy for the saved-rank fixtures."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import prepare_assets
import retrieval


def cli(*args):
    with patch.object(sys, 'argv', ['retrieval.py', *args]):
        retrieval.main()


class MatrixTests(unittest.TestCase):
    def test_plan_and_matrix_select_the_same_cases(self):
        selected = ['--dataset', 'flickr30k', '--method', 'ceta', '--source', 'albef', '--target', 'tcl']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(retrieval.subprocess, 'run') as launch:
                cli('matrix', '--output', tmp)
                self.assertEqual(launch.call_count, 160)
                for call in launch.call_args_list[1::2]:
                    self.assertEqual(call.args[0][-3:], ['evaluate', '--target', 'all'])
                launch.reset_mock()
                cli('matrix', '--output', tmp, *selected)
                self.assertEqual(launch.call_count, 2)
                case = root/'flickr30k/ceta/albef'
                self.assertIn(str(case), launch.call_args_list[0].args[0])
                self.assertEqual(launch.call_args.args[0][-3:], ['evaluate', '--target', 'tcl'])
                launch.reset_mock()
                case.mkdir(parents=True)
                (case/'generation.json').touch()
                cli('matrix', '--output', tmp, *selected)
                self.assertEqual(launch.call_count, 1)
        for args, expected in [((), (80, 320, 640)), (selected, (1, 1, 2))]:
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                cli('plan', *args)
            self.assertEqual(capture.getvalue().splitlines()[-3:],
                             [f'Generations: {expected[0]}', f'Source-target evaluations: {expected[1]}',
                              f'TR/IR ASR@1 cells: {expected[2]}'])
        with patch.object(retrieval, 'run') as run:
            cli('generate')
            args = run.call_args.args[0]
            self.assertEqual((args.dataset, args.method, args.source), ('flickr30k', 'ceta', 'albef'))

    def test_previous_generator_identity(self):
        current = retrieval.identity(SimpleNamespace(dataset='flickr30k', method='ceta', source='albef', limit=4))
        previous = {**current, 'source_sha256': '40d9e26431c9f790e8463e8d12cc47aeb73fe9ce75201b8c6566b956c6250ec2'}
        self.assertTrue(retrieval.matches_generation(previous, current))
        for change in [dict(method='pgd'), dict(limit=None), dict(source_sha256='unknown')]:
            self.assertFalse(retrieval.matches_generation({**previous, **change}, current))
        self.assertFalse(retrieval.matches_generation(previous, {**current, 'source_sha256': 'unknown'}))


class EvaluationTests(unittest.TestCase):
    def test_completed_target_reuse_requires_identity_and_rank_consistency(self):
        case = SimpleNamespace(dataset='flickr30k', method='ceta', source='albef')
        with tempfile.TemporaryDirectory() as tmp:
            leaf = Path(tmp)
            np.savez(leaf/'clean.npz', text_retrieval=[0, 1], image_retrieval=[1, 1, 1])
            np.savez(leaf/'adversarial.npz', text_retrieval=[1, 0], image_retrieval=[0, 0, 1])
            rows = [dict(**vars(case), target='tcl', direction=d, k=1, clean_hits=n,
                         lost_clean_hits=n, asr_percent=100.0 if n else None) for d,n in [('tr', 1), ('ir', 0)]]

            def record(metrics):
                (leaf/'metrics.json').write_text(json.dumps(metrics))
                return dict(target='tcl', generation_sha256='gallery-id',
                            files={n:retrieval.sha256_file(leaf/n) for n in retrieval.EVALUATION_FILES})

            def verify(receipt):
                (leaf/'complete.json').write_text(json.dumps(receipt))
                retrieval.verify_evaluation(leaf, 'gallery-id', case, 'tcl', 2, 3)

            good = record(rows)
            verify(good)
            for change in [dict(target='albef'), dict(generation_sha256='other-gallery'), dict(files={}),
                           dict(files={**good['files'], '../other':'extra'}),
                           dict(files={**good['files'], 'metrics.json':'wrong'})]:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verify({**good, **change})
            for key, value in [('target', 'albef'), ('source', 'tcl'), ('dataset', 'mscoco'),
                               ('method', 'pgd'), ('direction', 'ir'), ('clean_hits', True),
                               ('lost_clean_hits', 0), ('asr_percent', 0.0)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    verify(record([{**rows[0], key:value}, rows[1]]))
            with self.assertRaises(ValueError):
                verify(record([rows[0], {**rows[1], 'asr_percent':0}]))
            np.savez(leaf/'clean.npz', text_retrieval=[3, 1], image_retrieval=[1, 1, 1])
            with self.assertRaises(ValueError):
                verify(record(rows))


class ImageTests(unittest.TestCase):
    def test_existing_images_match_archive_and_extras_are_ignored(self):
        for kind in ('zip', 'tar'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = [{'image':'val2014/a.jpg', 'caption':['caption']}]
                annotation = root/'split.json'
                annotation.write_text(json.dumps(rows))
                spec = dict(annotation='split.json', image_root='images', images=1, captions=1,
                            annotation_sha256=prepare_assets.digest(annotation),
                            ordered_test_tree_sha256=hashlib.sha256(
                                f'val2014/a.jpg:4:{hashlib.sha256(b"good").hexdigest()}'.encode()
                            ).hexdigest())
                protocol = {'datasets':{'mscoco':spec}}
                (root/'configs').mkdir()
                (root/'configs/main.json').write_text(json.dumps(protocol))
                archive = root/('original.'+kind)
                entries = [('previews/a.jpg', b'bad'), ('wrapper/val2014/a.jpg', b'good'), ('other.jpg', b'no')]
                if kind == 'zip':
                    with zipfile.ZipFile(archive, 'w') as handle:
                        for name, data in entries:
                            handle.writestr(name, data)
                else:
                    with tarfile.open(archive, 'w') as handle:
                        for name, data in entries:
                            member = tarfile.TarInfo(name)
                            member.size = len(data)
                            handle.addfile(member, io.BytesIO(data))
                with (patch.object(prepare_assets, 'ROOT', root), patch.object(retrieval, 'ROOT', root),
                      patch.object(retrieval, 'PROTOCOL', protocol)):
                    prepare_assets.install_images('mscoco', archive)
                    prepare_assets.install_images('mscoco', archive)
                    dest = root/'images/val2014/a.jpg'
                    dest.write_bytes(b'gone')
                    with self.assertRaisesRegex(ValueError, 'val2014/a.jpg'):
                        prepare_assets.install_images('mscoco', archive)
                    self.assertEqual(dest.read_bytes(), b'gone')
                    self.assertEqual(list((root/'images').rglob('*.jpg')), [dest])
