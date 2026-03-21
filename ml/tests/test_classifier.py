import numpy as np
import pytest


def test_invalid_model_type():
    from ml.classifier import StrokeClassifier

    with pytest.raises(ValueError, match="Unknown model_type"):
        StrokeClassifier("fake.pt", device="cpu", model_type="invalid")


def test_3dcnn_default():
    from ml.classifier import StrokeClassifier

    clf = StrokeClassifier("nonexistent.pt", device="cpu")
    assert clf.model_type == "3dcnn"


def test_swin_accepted():
    from ml.classifier import StrokeClassifier

    clf = StrokeClassifier("nonexistent.pt", device="cpu", model_type="swin")
    assert clf.model_type == "swin"


def test_classify_shape():
    from ml.classifier import StrokeClassifier, STROKE_CLASSES

    clf = StrokeClassifier("nonexistent.pt", device="cpu", model_type="3dcnn")
    clips = np.random.randint(0, 255, (2, 16, 64, 64, 3), dtype=np.uint8)
    results = clf.classify(clips)
    assert len(results) == 2
    for r in results:
        assert "stroke" in r
        assert "confidence" in r
