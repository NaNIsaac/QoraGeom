import pytest

from qorageom import DETECTORS, get_detector


def test_detector_registry():
    assert get_detector("34idcTIM2").dims == (512, 512)
    assert get_detector("e4m").pixel_size == (75e-6, 75e-6)
    assert get_detector("mpxgaas").orientation == ("x-", "y-")
    assert {"34idcTIM1", "34idcTIM2", "ASI", "mpxgaas", "e4m"} == set(DETECTORS)
    with pytest.raises(KeyError):
        get_detector("nope")
