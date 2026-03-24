import numpy as np
import pytest
import torch


@pytest.fixture
def dummy_3dcnn_weights(tmp_path):
    """Create a valid 3DCNN state dict file for testing."""
    from ml.classifier import Simple3DCNN, STROKE_CLASSES

    model = Simple3DCNN(num_classes=len(STROKE_CLASSES))
    path = tmp_path / "test_3dcnn.pt"
    torch.save(model.state_dict(), str(path))
    return str(path)


@pytest.fixture
def dummy_swin_weights(tmp_path):
    """Create a valid VideoSwinTransformer state dict file for testing."""
    from ml.classifier import VideoSwinTransformer, STROKE_CLASSES

    model = VideoSwinTransformer(num_classes=len(STROKE_CLASSES))
    path = tmp_path / "test_swin.pt"
    torch.save(model.state_dict(), str(path))
    return str(path)


def test_invalid_model_type():
    from ml.classifier import StrokeClassifier

    with pytest.raises(ValueError, match="Unknown model_type"):
        StrokeClassifier("fake.pt", device="cpu", model_type="invalid")


def test_missing_weights_raises():
    from ml.classifier import StrokeClassifier

    with pytest.raises(FileNotFoundError, match="weights not found"):
        StrokeClassifier("nonexistent.pt", device="cpu")


def test_3dcnn_default(dummy_3dcnn_weights):
    from ml.classifier import StrokeClassifier

    clf = StrokeClassifier(dummy_3dcnn_weights, device="cpu")
    assert clf.model_type == "3dcnn"


def test_swin_accepted(dummy_swin_weights):
    from ml.classifier import StrokeClassifier

    clf = StrokeClassifier(dummy_swin_weights, device="cpu", model_type="swin")
    assert clf.model_type == "swin"


def test_classify_shape(dummy_3dcnn_weights):
    from ml.classifier import StrokeClassifier, STROKE_CLASSES

    clf = StrokeClassifier(dummy_3dcnn_weights, device="cpu", model_type="3dcnn")
    clips = np.random.randint(0, 255, (2, 16, 64, 64, 3), dtype=np.uint8)
    results = clf.classify(clips)
    assert len(results) == 2
    for r in results:
        assert "stroke" in r
        assert "confidence" in r
