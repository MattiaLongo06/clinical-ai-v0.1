from PIL import Image
from src.inference import load_imagenet_resnet18, toy_prob_from_brightness
from src.preprocessing import prepare_input

def test_model_loads():
    model, weights = load_imagenet_resnet18()
    assert model is not None and weights is not None

def test_toy_probs_sum_to_one():
    img = Image.new("RGB", (50,50), color="black")
    p_normal, p_pneu, _ = toy_prob_from_brightness(img)
    assert abs((p_normal + p_pneu) - 1.0) < 1e-6
