import unittest

import torch

from moondream_mini import MiniConfig, MiniMoondream
from scripts.train import set_vision_trainable


class ModelAndTrainingTests(unittest.TestCase):
    def setUp(self):
        self.model = MiniMoondream(
            MiniConfig(
                vocab_size=32,
                image_size=32,
                patch_size=16,
                num_image_tokens=5,
                vision_dim=16,
                text_dim=16,
                num_heads=4,
                num_vision_layers=2,
                num_text_layers=1,
                ff_dim=32,
                max_text_len=16,
            )
        )

    def test_model_forward_shape(self):
        logits = self.model(torch.rand(2, 3, 32, 32), torch.ones(2, 3, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (2, 8, 32))

    def test_v6_vision_attention_compatibility_is_default(self):
        self.assertTrue(self.model.cfg.vision_is_causal)

    def test_unfreeze_last_layer_keeps_earlier_blocks_frozen(self):
        set_vision_trainable(self.model, False)
        set_vision_trainable(self.model, True, unfreeze_last_layer=True)
        self.assertFalse(any(p.requires_grad for p in self.model.vision.blocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.vision.blocks[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.vision.proj.parameters()))


if __name__ == "__main__":
    unittest.main()
