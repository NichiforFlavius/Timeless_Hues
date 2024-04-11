import logging
import time
from datetime import timedelta
from pathlib import Path

import cv2
import onnxruntime
import torch
import numpy as np
from PIL import Image
from rich.console import Console

CONSOLE = Console(color_system='truecolor')

logging.basicConfig(filename='model_loading.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - triggered by %(filename)s',
                    datefmt='%H:%M - %d %B %Y')


def load_onnx_model(model_path: str | Path):
    ...

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

    except onnxruntime.capi.onnxruntime_pybind11_state.InvalidProtobuf:
        CONSOLE.print(f'Error when loading model: [cyan]{model_path.as_posix()}[/cyan]')
        logging.error(f'Model \"{model_path.name}\" could not be loaded!')
    return None


def preprocess_image_for_inference(image, model_size):
    ...

    start_preprocess_time = time.monotonic()

    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    # Extract original size of the image
    height, width = img.shape[:2]

    # normalize image pixels
    img = (img / 255.0).astype(np.float32)

    # # extract the original L channel from the CIELAB color space
    # orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

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
    logging.info(f'Preprocessed the original image \"{image}\" ({width}x{height}) in {timedelta(seconds=start_preprocess_time - end_preprocess_time)},'
                 f' for model size of {model_size}x{model_size}')
    return tensor_gray_rgb


def run_inference(input_tensor, input_name, onnx_rt_session,
                  original_width, original_height, original_l_channel):

    inference_outputs = onnx_rt_session.run(None, {input_name: input_tensor.numpy()})
    output_data = inference_outputs[0]

    output_ab_resize = output_data[0].transpose(1, 2, 0)
    output_ab_resize = cv2.resize(output_ab_resize, (original_width, original_height))

    output_lab = np.concatenate((original_l_channel, output_ab_resize), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

    output_img = (output_bgr * 255.0).round().astype(np.uint8)
    return output_img


def test_inference_main():
    pass
    sess, input_name = load_onnx_model(r'F:\Facultate\Dizertatie\Proiect\models\large_model_simplified.onnx')
    tensor_rgb = preprocess_image_for_inference(r"C:\Users\flavi\Desktop\photo_2024-04-10_17-54-40.jpg", 512)
    img = cv2.imread(r"C:\Users\flavi\Desktop\photo_2024-04-10_17-54-40.jpg")
    h, w = img.shape[:2]
    img = (img / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    result = run_inference(tensor_rgb, input_name, sess, w, h, orig_l)
    pil_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    pil_image.save('test_pil.png')
    cv2.imwrite(f'test.png', result)

    # model_path = r'F:\Facultate\Dizertatie\Proiect\models\large_model_simplified.onnx'
    # onnx_rt_session = onnxruntime.InferenceSession(model_path)
    #
    # model_size = 512
    # img = cv2.imread(r"C:\Users\flavi\Desktop\bird-8017963_640.jpg")
    #
    # height, width = img.shape[:2]
    #
    # # normalize image pixels
    # img = (img / 255.0).astype(np.float32)
    #
    # # # extract the original L channel from the CIELAB color space
    # orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)
    #
    # # resize image to the size of the model's input
    # img = cv2.resize(img, (model_size, model_size))
    #
    # # extract the resized L channel from the CIELAB color space
    # img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
    #
    # # create the two missing channels as empty arrays to obtain a grayscale image
    # img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    #
    # # convert LAB representation to RGB representation
    # img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    #
    # # convert array image to input tensor for the model
    # tensor_gray_rgb = (
    #     torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
    #     .float()
    #     .unsqueeze(0)
    #     .to(torch.device("cpu"))
    # )
    # inference_outputs = onnx_rt_session.run(None, {'x.1': tensor_gray_rgb.numpy()})
    # output_data = inference_outputs[0]
    #
    # output_ab_resize = output_data[0].transpose(1, 2, 0)
    # output_ab_resize = cv2.resize(output_ab_resize, (width, height))
    #
    # output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
    # output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    #
    # output_img = (output_bgr * 255.0).round().astype(np.uint8)
    # cv2.imwrite(f'test.png', output_img)


if __name__ == "__main__":
    test_inference_main()
