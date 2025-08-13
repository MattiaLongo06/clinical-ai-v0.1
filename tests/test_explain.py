from PIL import Image
from src.inference import load_imagenet_resnet18
from src.preprocessing import prepare_input
from src.explain import gradcam_heatmap

def test_gradcam_runs():
    model, weights = load_imagenet_resnet18()
    img = Image.new("RGB", (224,224), color="white")
    x = prepare_input(img, weights)
    cam = gradcam_heatmap(model, x, target_layer="layer4")
    assert cam.ndim == 2
