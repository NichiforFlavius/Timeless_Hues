# importing modules
import io
from datetime import datetime

import streamlit as st
from PIL import ImageFilter, ImageEnhance

from ML_Side.load_onnx_model import *

logging.basicConfig(filename='app_running.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - triggered by %(filename)s',
                    datefmt='%H:%M - %d %B %Y')


def colorize_image(edited_img, image_name, enabled=False):
    if enabled:
        sess, input_name = load_onnx_model(r'F:\Facultate\Dizertatie\Proiect\models\large_model_simplified.onnx')
        image_cv2 = np.array(edited_img)
        h, w = image_cv2.shape[:2]
        tensor_rgb = preprocess_image_for_inference(image_cv2, 512)
        image_cv2 = (image_cv2 / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2Lab)[:, :, :1]
        result = run_inference(tensor_rgb, input_name, sess, w, h, orig_l)
        np.save(image_name, result, allow_pickle=True)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    else:
        return edited_img


def main():
    st.set_page_config(page_title="Image editing app")
    st.header("Timeless Hues 📸")
    st.subheader("Upload an image to get started")
    image = st.file_uploader("Upload an image", type=[
        "png", "jpg"], accept_multiple_files=False, )

    # if image uploaded
    if image:
        # getting image in PIL
        if Path(f'{image.name}.npy').exists():
            edited_img = Image.fromarray(cv2.cvtColor(np.load(f'{image.name}.npy'), cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(image)
            edited_img = img

        # adding sidebar
        st.sidebar.header("Editing panel")
        # writing settings code
        st.sidebar.write("Settings")
        setting_sharp = st.sidebar.slider("Sharpness", min_value=0.01, max_value=10.0, value=1.0)  # done
        setting_color = st.sidebar.slider("Color", min_value=0.01, max_value=10.0, value=1.0)  # done
        setting_brightness = st.sidebar.slider("Exposure", min_value=0.01, max_value=10.0, value=1.0)  # done
        setting_contrast = st.sidebar.slider("Contrast", min_value=0.01, max_value=10.0, value=1.0)  # done
        setting_flip_image = st.sidebar.selectbox("Flip Image", options=(
            "select flip direction", "FLIP_TOP_BOTTOM", "FLIP_LEFT_RIGHT"))
        colorize_button = st.button('Colorize')
        # writing filters code
        st.sidebar.write("Filters")
        filter_blur = st.sidebar.checkbox("Blur")

        if colorize_button:
            print('here')
            enable_colorize = True
        else:
            print('there')
            enable_colorize = False

        if filter_blur:
            filter_blur_strength = st.sidebar.slider("Select Blur strength", min_value=0.00, max_value=10.0, value=0.0)

        # checking setting_sharp value
        if setting_sharp:
            sharp_value = setting_sharp
        else:
            sharp_value = 0

        # checking color
        if setting_color:
            set_color = setting_color
        else:
            set_color = 1

        # checking brightness
        if setting_brightness:
            set_brightness = setting_brightness
        else:
            set_brightness = 1

        # checking contrast
        if setting_contrast:
            set_contrast = setting_contrast
        else:
            set_contrast = 1

        # checking setting_flip_image
        flip_direction = setting_flip_image

        # colorizing image:
        with st.spinner('Colorizing...'):
            edited_img = colorize_image(edited_img, image.name, enable_colorize)

        # implementing sharpness
        sharp = ImageEnhance.Sharpness(edited_img)
        edited_img = sharp.enhance(sharp_value)

        # implementing colors
        color = ImageEnhance.Color(edited_img)
        edited_img = color.enhance(set_color)

        # implementing brightness
        brightness = ImageEnhance.Brightness(edited_img)
        edited_img = brightness.enhance(set_brightness)

        # implementing contrast
        contrast = ImageEnhance.Contrast(edited_img)
        edited_img = contrast.enhance(set_contrast)

        # implementing flip direction
        if flip_direction == "FLIP_TOP_BOTTOM":
            edited_img = edited_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif flip_direction == "FLIP_LEFT_RIGHT":
            edited_img = edited_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            pass

        if filter_blur:
            if filter_blur_strength:
                set_blur = filter_blur_strength
                edited_img = edited_img.filter(ImageFilter.GaussianBlur(set_blur))

        # displaying edited image

        st.image(edited_img)

        img_bytes = io.BytesIO()
        edited_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        st.download_button('Download image', data=img_bytes, file_name=f'revived_image_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.png', mime='image/png')
        if st.button('Clear cache'):
            if Path(f'{image.name}.npy').exists():
                Path(f'{image.name}.npy').unlink(missing_ok=True)


if __name__ == "__main__":
    main()
