from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import Batch
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModel, InnerModelConfig


def real_denoiser(ttc_weight, dense_dim=0):
    torch.manual_seed(0)
    inner = InnerModelConfig(img_channels=3, num_steps_conditioning=2, cond_channels=8, depths=[1, 1, 1, 1],
                             channels=[8, 8, 8, 8], attn_depths=[0, 0, 0, 0], num_actions=5,
                             dense_bottleneck_dim=dense_dim, img_size=[16, 32])
    denoiser = Denoiser(DenoiserConfig(inner, 0.15, 0.0, pixel_mask_gain=0.0, pixel_mae_weight=1.0,
                                       diffusion_mse_weight=0.0, ttc_loss_weight=ttc_weight, ttc_hidden_dim=16,
                                       ttc_dropout=0.0))
    denoiser.setup_training(SigmaDistributionConfig(-0.4, 1.2, 2e-3, 20))
    generator = torch.Generator().manual_seed(1)
    for p in denoiser.inner_model.parameters():  # The default init zeroes layers, which hides gradients.
        p.data = torch.randn(p.shape, generator=generator) * 0.1
    return denoiser


def batch(ttc=None, mask=None):
    generator = torch.Generator().manual_seed(2)
    b, t = 2, 3  # 2 context frames + 1 target
    info = [{} if ttc is None else {"obs_ttc": row} for row in (ttc if ttc is not None else [None] * b)]
    return Batch(obs=torch.rand(b, t, 3, 16, 32, generator=generator) * 2 - 1, act=torch.zeros(b, t, dtype=torch.long),
                 rew=torch.zeros(b, t), end=torch.zeros(b, t, dtype=torch.long), trunc=torch.zeros(b, t, dtype=torch.long),
                 mask_padding=torch.ones(b, t, dtype=torch.bool) if mask is None else mask, info=info, segment_ids=[])


def gradients(weight, data, dense_dim=0):
    denoiser = real_denoiser(weight, dense_dim)
    torch.manual_seed(3)
    loss, logs = denoiser(data)
    loss.backward()
    return denoiser, logs, {n: p.grad.clone() for n, p in denoiser.named_parameters() if p.grad is not None}


class JointTtcTests(unittest.TestCase):
    def test_ttc_gradient_reaches_the_down_path_and_bottleneck_but_not_the_up_path(self):
        data = batch(ttc=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])])
        _, _, plain = gradients(0.0, data)
        denoiser, logs, joint = gradients(1.0, data)
        self.assertIn("loss_ttc", logs)
        self.assertTrue(any(n.startswith("ttc_head.") for n in joint))
        for name, grad in plain.items():
            if name.startswith(("inner_model.unet.u_blocks", "inner_model.unet.upsamples", "inner_model.conv_out")):
                self.assertTrue(torch.allclose(grad, joint[name], atol=1e-6), name)
        for prefix in ("inner_model.conv_in", "inner_model.unet.d_blocks", "inner_model.unet.mid_blocks"):
            changed = [n for n in joint if n.startswith(prefix) and not torch.allclose(plain[n], joint[n], atol=1e-6)]
            self.assertTrue(changed, f"the TTC loss should change gradients under {prefix}")

    def test_padded_targets_are_ignored(self):
        ttc = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 1e6])]
        mask = torch.ones(2, 3, dtype=torch.bool)
        _, counted, _ = gradients(1.0, batch(ttc, mask))
        mask[1, 2] = False
        _, skipped, _ = gradients(1.0, batch(ttc, mask))
        self.assertGreater(counted["loss_ttc"].item(), 1e4)
        self.assertLess(skipped["loss_ttc"].item(), 100)

    def test_off_by_default_and_needs_obs_ttc_when_on(self):
        self.assertFalse(any("ttc_head" in key for key in real_denoiser(0.0).state_dict()))
        with self.assertRaisesRegex(ValueError, "obs_ttc"):
            real_denoiser(1.0)(batch())


class DenseBottleneckTests(unittest.TestCase):
    def test_size_and_starts_as_the_plain_unet(self):
        torch.manual_seed(0)
        plain = InnerModel(InnerModelConfig(3, 2, 8, [1, 1, 1, 1], [8, 8, 8, 8], [0, 0, 0, 0], 5))
        dense = InnerModel(InnerModelConfig(3, 2, 8, [1, 1, 1, 1], [8, 8, 8, 8], [0, 0, 0, 0], 5,
                                            dense_bottleneck_dim=16, img_size=[16, 32]))
        features = 8 * 2 * 4  # 16×32 halved three times
        added = sum(p.numel() for p in dense.parameters()) - sum(p.numel() for p in plain.parameters())
        self.assertEqual(added, 2 * features * 16 + features + 16)
        dense.load_state_dict(plain.state_dict(), strict=False)
        for p in plain.parameters():  # conv_out starts at zero, which would hide any difference.
            p.data.normal_(0, 0.1)
        dense.load_state_dict(plain.state_dict(), strict=False)
        args = (torch.randn(2, 3, 16, 32), torch.zeros(2), torch.randn(2, 6, 16, 32), torch.zeros(2, 2, dtype=torch.long))
        self.assertTrue(torch.allclose(plain(*args), dense(*args)))
        self.assertEqual(dense.extract_latent(args[2], args[3]).shape, (2, 16))

    def test_ttc_head_reads_the_dense_vector_and_trains_dense_in_not_dense_out(self):
        data = batch(ttc=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])])
        _, _, plain = gradients(0.0, data, dense_dim=16)
        denoiser, _, joint = gradients(1.0, data, dense_dim=16)
        self.assertEqual(denoiser.ttc_head.head[0].in_features, 16)
        for name in ("inner_model.unet.dense_out.weight", "inner_model.unet.dense_out.bias"):
            self.assertTrue(torch.allclose(plain[name], joint[name], atol=1e-6), name)
        self.assertFalse(torch.allclose(plain["inner_model.unet.dense_in.weight"],
                                        joint["inner_model.unet.dense_in.weight"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
