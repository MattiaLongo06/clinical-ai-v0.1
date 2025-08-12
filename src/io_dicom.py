import pydicom
import numpy as np
from PIL import Image

def load_dicom_to_pil(file_obj) -> Image.Image:
    """
    Carica un file DICOM, applica windowing semplice e converte in immagine PIL RGB.
    :param file_obj: percorso del file o file-like object (.dcm)
    :return: immagine PIL RGB
    """
    # Leggi il DICOM
    dicom_data = pydicom.dcmread(file_obj)

    # Ottieni pixel array
    pixel_array = dicom_data.pixel_array.astype(np.float32)

    # Applica rescale slope/intercept se presenti
    intercept = getattr(dicom_data, "RescaleIntercept", 0.0)
    slope = getattr(dicom_data, "RescaleSlope", 1.0)
    pixel_array = pixel_array * slope + intercept

    # Windowing (se disponibile nei metadati)
    if hasattr(dicom_data, "WindowCenter") and hasattr(dicom_data, "WindowWidth"):
        center = dicom_data.WindowCenter
        width = dicom_data.WindowWidth

        # A volte sono array, prendiamo il primo elemento
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]

        min_val = center - width / 2
        max_val = center + width / 2
        pixel_array = np.clip(pixel_array, min_val, max_val)
    else:
        # Normalizzazione base
        min_val = np.min(pixel_array)
        max_val = np.max(pixel_array)
        pixel_array = np.clip(pixel_array, min_val, max_val)

    # Riscalamento in 0-255 per visualizzazione
    pixel_array -= np.min(pixel_array)
    pixel_array /= np.max(pixel_array)
    pixel_array *= 255.0

    # Converti in uint8 e poi in RGB
    image_8bit = pixel_array.astype(np.uint8)
    pil_img = Image.fromarray(image_8bit).convert("RGB")

    return pil_img
