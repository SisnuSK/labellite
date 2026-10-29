import numpy as np
import timm
import torch
from torch import nn

from labellite.explain.gallery import pick, summarize_review
from labellite.explain.gradcam import GradCAM, ProbeHead
from labellite.train.probe import softmax


def test_gradcam_on_cnn_highlights_the_evidence_channel():
    # Grad-CAM weights channels by their mean gradient. Channel 0 (bright top-left) drives the
    # class-1 logit; channel 1 (bright bottom-right) is ignored. The map should follow channel 0.
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Identity()

        def forward(self, x):
            a = self.feat(x)
            q = a[:, 0].mean((1, 2))
            return torch.stack([-q, q], 1)

    x = torch.zeros(2, 2, 8, 8)
    x[:, 0, :4, :4] = 1.0
    x[:, 1, 4:, 4:] = 1.0
    model = Toy()
    cam = GradCAM(model, model.feat)
    maps, logits = cam(x, classes=torch.tensor([1, 1]))
    assert maps.shape == (2, 8, 8) and logits.shape == (2, 2)
    assert maps.max() <= 1 and maps.min() >= 0
    assert np.allclose(maps[:, :4, :4], 1) and np.allclose(maps[:, 4:, 4:], 0)


def test_gradcam_on_vit_tokens_and_probe_head():
    torch.manual_seed(0)
    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, img_size=32)
    vit.eval().requires_grad_(False)
    rng = np.random.default_rng(0)
    head = ProbeHead(vit, rng.normal(size=(1, vit.num_features)), np.zeros(1))
    cam = GradCAM(head, vit.blocks[-1].norm1, vit.num_prefix_tokens)
    torch.manual_seed(0)
    maps, logits = cam(torch.rand(3, 3, 32, 32), out_size=20)
    assert maps.shape == (3, 20, 20) and np.isfinite(maps).all()
    assert logits.shape == (3, 2) and (logits[:, 0] == 0).all()  # binary -> [0, z]
    # The class logit depends on the hooked patch tokens (it would not on the last block's
    # output tokens, which the CLS feature never sees).
    x = torch.rand(3, 3, 32, 32, requires_grad=True)
    z = head(x)
    (g,) = torch.autograd.grad(z[:, 1].sum(), cam._acts)
    assert g[:, vit.num_prefix_tokens :].abs().sum() > 0


def test_pick_orders_cases():
    y = np.array([0, 0, 1, 1])
    probs = softmax(np.array([[3.0, 0], [0, 1.0], [0.1, 0], [0, 5.0]]))
    np.testing.assert_array_equal(pick(probs, y, "confident_errors", 2), [1, 2])
    np.testing.assert_array_equal(pick(probs, y, "confident_correct", 1), [3])
    np.testing.assert_array_equal(pick(probs, y, "uncertain", 1), [2])


def test_review_summary(tmp_path):
    sheet = tmp_path / "review_sample.csv"
    sheet.write_text(
        "tile,id,true,pred,confidence,failure_type,notes\n"
        "a,1,x,y,0.9,ambiguous,\nb,2,x,y,0.8,ambiguous,\nc,3,x,y,0.7,label_noise,\nd,4,x,y,0.6,,\n"
    )
    out = summarize_review(sheet)
    assert "Reviewed 3 of 4" in out and "| ambiguous | 2 | 67% |" in out


def test_binary_probe_explains_both_classes():
    # With logits [0, z], the raw class-0 logit has no gradient; the log-probability does.
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Identity()

        def forward(self, x):
            z = self.feat(x)[:, 0].mean((1, 2))
            return torch.stack([torch.zeros_like(z), -z], 1)  # class 0 wins when z > 0

    x = torch.zeros(1, 2, 8, 8)
    x[:, 0, :4, :4] = 1.0
    model = Toy()
    maps, logits = GradCAM(model, model.feat)(x)
    assert logits.argmax() == 0 and np.allclose(maps[0, :4, :4], 1)
