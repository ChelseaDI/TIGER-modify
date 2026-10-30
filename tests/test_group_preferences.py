import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import T5Config, AutoModelForSeq2SeqLM
from build_group_features import build_features
from group_preference import GroupPreference
from preference_t5 import PreferenceT5
from utils import RecDataset


class PreferenceTests(unittest.TestCase):
    def test_residuals_and_sample_history(self):
        indices = {str(i): ['<a_0>', '<b_0>', '<c_0>', '<d_{}>'.format(i)]
                   for i in range(5)}
        features = build_features(torch.arange(10).float().reshape(5, 2), indices)
        self.assertTrue(torch.allclose(features['residuals'].sum(0), torch.zeros(2)))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'features.pt')
            torch.save(features, path)
            args = SimpleNamespace(dataset='synthetic', tokenizer_plm='sentence-t5-base',
                                   K=256, D=3, add_user_prefix=False, item_sep=',', max_len=2,
                                   token_type='sid_nc', use_group_preferences=True,
                                   group_features=path, group_prefix_len=2)
            def load(ds):
                ds.indices = indices
                ds.inter_data = {'u': list(range(5))}
            with patch.object(RecDataset, '_load_data', load):
                train = RecDataset(args, 'train')
                valid = RecDataset(args, 'valid')
                test = RecDataset(args, 'test')
            for sample, ids in [(train[0], [0]), (train[1], [0, 1]),
                                (valid[0], [1, 2]), (test[0], [2, 3])]:
                self.assertTrue(torch.allclose(sample['group_preferences'],
                                               features['residuals'][ids].mean(0, keepdim=True)))
            with self.assertRaises(ValueError):
                GroupPreference(path, indices, 1)

    def test_gradient_padding_generation_and_reload(self):
        config = T5Config(vocab_size=24, d_model=16, d_ff=32, num_layers=1,
                          num_decoder_layers=1, num_heads=2, dropout_rate=0,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        config.content_dim = 4
        model = PreferenceT5(AutoModelForSeq2SeqLM.from_config(config), config.content_dim)
        ids = torch.tensor([[2, 3, 1], [4, 1, 0]])
        mask = (ids != 0).long()
        preferences = torch.randn(2, 2, 4)
        pmask = torch.tensor([[1, 1], [1, 0]])
        labels = torch.tensor([[5, 1], [6, 1]])
        loss = model(input_ids=ids, attention_mask=mask, group_preferences=preferences,
                     preference_mask=pmask, labels=labels).loss
        loss.backward()
        self.assertGreater(model.preference_projection.weight.grad.abs().sum().item(), 0)
        model.eval()
        encoded, full_mask = model.encode_preferences(ids, mask, preferences, pmask)
        changed = preferences.clone()
        changed[1, 1] = 1000
        other, _ = model.encode_preferences(ids, mask, changed, pmask)
        out = model.t5(encoder_outputs=encoded, attention_mask=full_mask, labels=labels).logits
        out2 = model.t5(encoder_outputs=other, attention_mask=full_mask, labels=labels).logits
        self.assertTrue(torch.allclose(out, out2, atol=1e-5))
        direct = model(input_ids=ids, attention_mask=mask,
                       group_preferences=preferences, preference_mask=pmask, labels=labels)
        self.assertTrue(torch.allclose(direct.logits, out, atol=1e-5))
        generated = model.generate(encoder_outputs=encoded, attention_mask=full_mask,
                                   num_beams=2, max_new_tokens=3)
        self.assertEqual(generated.shape[0], 2)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = PreferenceT5.from_pretrained(directory)
            self.assertTrue(torch.equal(model.preference_projection.weight,
                                        restored.preference_projection.weight))


if __name__ == '__main__':
    unittest.main()
