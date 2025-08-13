from PIL import Image
import numpy as np
from src.preprocessing import to_numpy, prepare_input

def test_to_numpy_shape():
    img = Image.new("RGB", (100, 60), color="white")
    arr = to_numpy(img, size=(64,64))
    assert arr.shape == (64,64,3)
    assert arr.dtype == np.float32 or arr.dtype == np.float64

def test_prepare_input_tensor():
    img = Image.new("RGB", (100, 60), color="white")
    x = prepare_input(img, weights=None)
    assert tuple(x.shape) == (1,3,224,224)
