import numpy as np

from labellite.data.datasets import PCam
from labellite.data.shift import StainShifted, stain_shift


def test_severity_zero_is_identity_and_round_trip_is_exact():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    np.testing.assert_array_equal(stain_shift(img, 0, 0.15, 0.25), img)
    # The OD -> stain -> OD round trip without a shift reproduces every pixel value.
    np.testing.assert_array_equal(stain_shift(img, 1, 0.0, 0.0), img)


def test_shift_grows_with_severity():
    rng = np.random.default_rng(1)
    img = rng.integers(60, 230, (32, 32, 3), dtype=np.uint8)
    diffs = [np.abs(stain_shift(img, s, 0.15, 0.25).astype(int) - img).mean() for s in (1, 2, 3)]
    assert 0 < diffs[0] < diffs[1] < diffs[2]


def test_stain_shifted_adapter(fake_pcam):
    base = PCam(fake_pcam)
    shifted = StainShifted(base, 2, 0.15, 0.25)
    assert shifted.name == "pcam_stain_s2_h0.15_e0.25" and shifted.class_names == base.class_names
    ids = ["test/3", "test/0"]
    arr = shifted.load_array(ids)
    np.testing.assert_array_equal(arr, stain_shift(base.load_array(ids), 2, 0.15, 0.25))
    np.testing.assert_array_equal(np.asarray(shifted.load_image("test/3")), arr[0])
    assert shifted.index().equals(base.index())
