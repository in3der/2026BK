import torch
import unittest

from model.speaker_behavior_encoder import SpeakerBehaviorEncoder


def _model():
    torch.manual_seed(0)
    model = SpeakerBehaviorEncoder(
        device="cpu", app_num_layers=1, app_mlp_dim=128, app_max_len=64
    )
    model.eval()
    return model


class SBEContractTest(unittest.TestCase):
    def test_temporal_contract_and_padding_invariance(self):
        model = _model()
        mel = torch.randn(2, 40, 80)
        dmm = torch.randn(2, 12, 486)
        au = torch.randn(2, 12, 25)
        mel_len = torch.tensor([40, 26])
        dmm_len = torch.tensor([12, 8])
        au_len = torch.tensor([12, 8])

        with torch.no_grad():
            result = model(
                mel, dmm, au,
                mel_len=mel_len, dmm_len=dmm_len, au_len=au_len,
                return_branches=True, return_mask=True,
            )
        c, e_aud, e_app, e_emo, mask = result
        self.assertEqual(c.shape, (2, 11, 512))
        self.assertEqual(e_aud.shape, c.shape)
        self.assertEqual(e_app.shape, c.shape)
        self.assertEqual(e_emo.shape, c.shape)
        self.assertEqual(mask.shape, (2, 11))
        self.assertEqual(mask.sum(dim=1).tolist(), [11, 7])

        mel_changed, dmm_changed, au_changed = mel.clone(), dmm.clone(), au.clone()
        mel_changed[1, 26:] = 1000
        dmm_changed[1, 8:] = -1000
        au_changed[1, 8:] = 500
        with torch.no_grad():
            c_changed, mask_changed = model(
                mel_changed, dmm_changed, au_changed,
                mel_len=mel_len, dmm_len=dmm_len, au_len=au_len,
                return_mask=True,
            )
        self.assertTrue(torch.equal(mask, mask_changed))
        self.assertTrue(torch.allclose(c[1, :7], c_changed[1, :7], atol=1e-5, rtol=1e-5))
        self.assertEqual(torch.count_nonzero(c[1, 7:]).item(), 0)

    def test_current_3dmm_projection_contract(self):
        model = _model()
        weight = model.app_encoder.transformer.embed_layer.weight
        self.assertEqual(tuple(weight.shape), (512, 486))


if __name__ == "__main__":
    unittest.main()
