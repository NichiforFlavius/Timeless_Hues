from PIL import Image

from ML_Side.load_onnx_model import *

CACHE_DIR = Path.cwd().joinpath('cache_files')
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_image_components(cv2_image: np.array):
    """
        Extract image width/height and greyscale (L from CIELAB color space)
        from the original image.

        Args:
            cv2_image (np.array): the input image as a numpy array/
        Returns:
            tuple(int, int, np.array): a tuple containing the width, height and greyscale channel of the image.
    """

    image_cv2 = np.array(cv2_image)
    height, width = image_cv2.shape[:2]
    image_cv2 = (image_cv2 / 255.0).astype(np.float32)
    l_channel = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2Lab)[:, :, :1]
    return width, height, l_channel


def colorize_image(model_path, model_size, edited_img, image_name, enabled=False):
    """
        Colorizes greyscale images using a dedicated pre-trained model.

        Args:
            model_path (str | Path): path to model in onnx format
            model_size (int): size in pixels for the model of the input (e.g. 512 -> 512x512 image)
            edited_img (np.array): image in numpy array format
            image_name (str): the name of the original image file.
            enabled (bool): flag that enables the colorizing process. Default is False.
        Returns:
            (np.array): the colorized image as a numpy array.
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
        np.save(CACHE_DIR / Path(image_name).stem, result, allow_pickle=True)

        return Image.fromarray(result)
    else:
        return edited_img
