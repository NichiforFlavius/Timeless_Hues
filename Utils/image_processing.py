

import cv2
import numpy as np
from PIL import Image

from ML_Side.load_onnx_model import *

CACHE_DIR = Path.cwd().joinpath('cache_files')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_image_components(cv2_image: np.array):
    """

    Args:
        cv2_image:

    Returns:

    """

    image_cv2 = np.array(cv2_image)
    height, width = image_cv2.shape[:2]
    image_cv2 = (image_cv2 / 255.0).astype(np.float32)
    l_channel = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2Lab)[:, :, :1]
    return width, height, l_channel


def colorize_image(model_path, model_size, edited_img, image_name, enabled=False):
    """

    Args:
        model_path:
        model_size:
        edited_img:
        image_name:
        enabled:

    Returns:

    """

    if enabled:
        # initialize onnx session with given model
        onnx_session, input_name = load_onnx_model(model_path)
        # load image
        image_cv2 = np.array(edited_img)
        # preprocess image to required model input specifications
        tensor_rgb = preprocess_image_for_inference(image_cv2, model_size)
        # extract original image components
        width, height, cielab_l_channel = extract_image_components(image_cv2)
        # run inference on the image
        result = run_inference(tensor_rgb, input_name, onnx_session, width, height, cielab_l_channel)
        # cache result locally for future loads
        np.save(CACHE_DIR / image_name, result, allow_pickle=True)

        return Image.fromarray(result)
    else:
        return edited_img
