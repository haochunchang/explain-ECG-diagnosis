from interpret import Explainer
from utils import preprocess_signals


def test_GradCamExplainer(model, const, helpers):

    data = helpers.simulate_ecg_signal(duration=const["DURATION"])
    sample = preprocess_signals(data["signal"])
    assert sample.shape == (1, 15, 2048)
    assert sample.requires_grad

    GradCamExplainer = Explainer(
        "GradCam",
        model=model.network,
        feature_module=model.network[:9],
        target_layer_names=["8"]
    )
    cam_mask = GradCamExplainer.explain_instance(sample)
    assert cam_mask.shape == (1, const["NUM_CHANNEL"], const["DURATION"])


def test_LimeExplainer(model, const, helpers):

    data = helpers.simulate_ecg_signal(duration=const["DURATION"])
    signal = data["signal"]
    label = data["label"].argmax()

    LimeExplainer = Explainer("LIME", model=model)
    explanation = LimeExplainer.explain_instance(signal, labels=[label])
    sample, mask = explanation.get_instance_and_mask(label)
    assert sample.shape == mask.shape
    assert sample.shape == (const["NUM_CHANNEL"], const["DURATION"])
