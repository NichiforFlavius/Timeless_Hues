import logging
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import torch
from rich.console import Console

CONSOLE = Console(color_system='truecolor')
LOGGING_DIR = Path.cwd().joinpath('logs')
LOGGING_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOGGING_DIR / 'app_events_history.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - triggered by %(filename)s',
                    datefmt='%H:%M - %d %B %Y')


def load_onnx_model(model_path: str | Path):
    """
        Loads the model using an ONNX session and returns it alongside the
        input name of the computational graph.

        Args:
            model_path (str | Path): path to the model in ONNX format.
        Returns:
            tuple(onnxruntime.InferenceSession, str): the onnxruntime session that contains the model + the input name of the computational graph.
    """

    if isinstance(model_path, str):
        model_path = Path(model_path)

    try:
        start_model_loading_time = time.monotonic()
        onnx_rt_session = onnxruntime.InferenceSession(model_path)
        end_model_loading_time = time.monotonic()

        input_name = onnx_rt_session.get_inputs()[0].name

        logging.info(f'Model \"{model_path.name}\" was loaded successfully in {timedelta(seconds=(end_model_loading_time - start_model_loading_time))}'
                     f'! It has the following input name: {input_name}')
        return onnx_rt_session, input_name

    except onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf and onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile:
        CONSOLE.print(f'Error when loading model: [cyan]{model_path.as_posix()}[/cyan]')
        logging.error(f'Model \"{model_path.name}\" could not be loaded!')
    return None


def preprocess_image_for_inference(image, model_size: int) -> torch.Tensor:
    """
        Extract and process all the information needed from the image
        to bring it into an inference-ready format.

        Args:
            image (str | np.array): either the image path or the image as an array.
            model_size (int): the shape of the input used with the current model (e.g. 512, 256, 128)
        Returns:
            (torch.Tensor): the image converted in the format expected by the model to run inference on it.
    """

    start_preprocess_time = time.monotonic()

    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    # Extract original size of the image
    height, width = img.shape[:2]

    # normalize image pixels
    img = (img / 255.0).astype(np.float32)

    # resize image to the size of the model's input
    img = cv2.resize(img, (model_size, model_size))

    # extract the resized L channel from the CIELAB color space
    img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]

    # create the two missing channels as empty arrays to obtain a grayscale image
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)

    # convert LAB representation to RGB representation
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    # convert array image to input tensor for the model
    tensor_gray_rgb = (
        torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
        .float()
        .unsqueeze(0)
        .to(torch.device("cpu"))
    )
    end_preprocess_time = time.monotonic()
    logging.info(f'Preprocessed the original image ({width}x{height}) in {timedelta(seconds=end_preprocess_time - start_preprocess_time)},'
                 f' for model size of {model_size}x{model_size}')
    return tensor_gray_rgb


def run_inference(input_tensor: torch.Tensor, input_name: str, onnx_rt_session: onnxruntime.InferenceSession,
                  original_width: int, original_height: int, original_l_channel: np.array) -> np.array:
    """
        Runs inference using an ONNX session and a given image in expected Tensor format,
        and returns the result as a numpy array.

        Args:
            input_tensor (torch.Tensor): the input image in tensor format to be used in the inference
            input_name (str): the input name of the computational graph
            onnx_rt_session (onnxruntime.InferenceSession): the ONNX session with the loaded model that will run the inference
            original_width (int): the original width of the image in pixels
            original_height (height): the original height of the image in pixels
            original_l_channel (np.array): the original greyscale channel (CIELAB color space) of the image.
        Returns:
            (np.array): the colorized image as a numpy array.
    """

    inference_outputs = onnx_rt_session.run(None, {input_name: input_tensor.numpy()})
    output_data = inference_outputs[0]

    output_ab_resize = output_data[0].transpose(1, 2, 0)
    output_ab_resize = cv2.resize(output_ab_resize, (original_width, original_height))

    output_lab = np.concatenate((original_l_channel, output_ab_resize), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

    output_img_bgr = (output_bgr * 255.0).round().astype(np.uint8)
    output_img = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
    return output_img
