import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import setup_iris


class IrisSetupTests(unittest.TestCase):
    def test_existing_verified_files_and_preserved_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, image, encoder = (root / name for name in ('sd-cli', 'iris.gguf', 'qwen.gguf'))
            runner.write_bytes(b'runner')
            if os.name != 'nt':
                runner.chmod(0o755)
            image.write_bytes(b'iris')
            encoder.write_bytes(b'qwen')
            iris = (*setup_iris.IRIS[:3], hashlib.sha256(b'iris').hexdigest())
            qwen = (*setup_iris.QWEN[:3], hashlib.sha256(b'qwen').hexdigest())
            config = root / 'config' / 'image-generator.json'
            with patch.object(setup_iris, 'IRIS', iris), patch.object(setup_iris, 'QWEN', qwen):
                setup_iris.prepare(runner, config, image_model=image, text_encoder=encoder)
                settings = json.loads(config.read_text(encoding='utf-8'))
                self.assertEqual(settings['diffusion_model'], str(image.resolve()))
                self.assertEqual(settings['text_encoder'], str(encoder.resolve()))
                self.assertNotIn('vae', settings)
                before = config.read_bytes()
                with self.assertRaises(FileExistsError):
                    setup_iris.prepare(runner, config, image_model=image, text_encoder=encoder)
                self.assertEqual(config.read_bytes(), before)

    def test_graphics_memory_budget_is_recorded_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, image, encoder = (root / name for name in ('sd-cli', 'iris.gguf', 'qwen.gguf'))
            runner.write_bytes(b'runner')
            if os.name != 'nt':
                runner.chmod(0o755)
            image.write_bytes(b'iris')
            encoder.write_bytes(b'qwen')
            iris = (*setup_iris.IRIS[:3], hashlib.sha256(b'iris').hexdigest())
            qwen = (*setup_iris.QWEN[:3], hashlib.sha256(b'qwen').hexdigest())
            with patch.object(setup_iris, 'IRIS', iris), patch.object(setup_iris, 'QWEN', qwen):
                for bad in (0.5, 65, float('nan'), float('inf'), 'low', True):
                    with self.assertRaisesRegex(ValueError, 'budget'):
                        setup_iris.prepare(runner, root / 'bad.json', image_model=image, text_encoder=encoder,
                                           max_vram_gib=bad)
                    self.assertFalse((root / 'bad.json').exists())
                setup_iris.prepare(runner, root / 'low.json', image_model=image, text_encoder=encoder, max_vram_gib=2)
                self.assertEqual(json.loads((root / 'low.json').read_text(encoding='utf-8'))['max_vram_gib'], 2)
                setup_iris.prepare(runner, root / 'default.json', image_model=image, text_encoder=encoder)
                self.assertNotIn('max_vram_gib', json.loads((root / 'default.json').read_text(encoding='utf-8')))
                setup_iris.prepare(runner, root / 'auto.json', image_model=image, text_encoder=encoder,
                                   max_vram_gib='auto')
                self.assertEqual(json.loads((root / 'auto.json').read_text(encoding='utf-8'))['max_vram_gib'], 'auto')
                self.assertNotIn('mmap', json.loads((root / 'default.json').read_text(encoding='utf-8')))
                for good in ('auto', True, False):
                    setup_iris.prepare(runner, root / f'mmap-{good}.json', image_model=image, text_encoder=encoder, mmap=good)
                    self.assertEqual(json.loads((root / f'mmap-{good}.json').read_text(encoding='utf-8'))['mmap'], good)
                for bad in ('on', 1, 0, 'yes'):
                    with self.assertRaisesRegex(ValueError, 'mmap'):
                        setup_iris.prepare(runner, root / 'bad-mmap.json', image_model=image, text_encoder=encoder, mmap=bad)
                    self.assertFalse((root / 'bad-mmap.json').exists())

    def test_mismatched_weights_do_not_create_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, image = root / 'sd-cli', root / 'wrong.gguf'
            runner.write_bytes(b'runner')
            if os.name != 'nt':
                runner.chmod(0o755)
            image.write_bytes(b'wrong')
            config = root / 'config.json'
            with self.assertRaisesRegex(ValueError, 'digest'):
                setup_iris.prepare(runner, config, image_model=image, text_encoder=image)
            self.assertFalse(config.exists())

    def test_downloads_pinned_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, image, encoder = (root / name for name in ('sd-cli', 'iris.gguf', 'qwen.gguf'))
            runner.write_bytes(b'runner')
            if os.name != 'nt':
                runner.chmod(0o755)
            image.write_bytes(b'iris')
            encoder.write_bytes(b'qwen')
            iris = (*setup_iris.IRIS[:3], hashlib.sha256(b'iris').hexdigest())
            qwen = (*setup_iris.QWEN[:3], hashlib.sha256(b'qwen').hexdigest())
            with patch.object(setup_iris, 'IRIS', iris), patch.object(setup_iris, 'QWEN', qwen), \
                 patch('huggingface_hub.hf_hub_download', side_effect=[str(image), str(encoder)]) as download:
                setup_iris.prepare(runner, root / 'config.json')
            self.assertEqual(download.call_args_list[0].kwargs,
                             {'repo_id': iris[0], 'filename': iris[1], 'revision': iris[2]})
            self.assertEqual(download.call_args_list[1].kwargs,
                             {'repo_id': qwen[0], 'filename': qwen[1], 'revision': qwen[2]})


if __name__ == '__main__':
    unittest.main()
