# importing modules
import io
import logging
from datetime import datetime

import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance

from ML_Side.load_onnx_model import *

logging.basicConfig(filename='app_running.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - triggered by %(filename)s',
                    datefmt='%H:%M - %d %B %Y')


def main():
    st.set_page_config(page_title="Image editing app")
    st.header("Timeless Hues 📸")
    st.subheader("Upload an image to get started")

    image = st.file_uploader("Upload an image", type=[
        "png", "jpg"], accept_multiple_files=False, )

    # if image uploaded
    if image:
        # getting image in PIL
        img = Image.open(image)
        # adding sidebar
        st.sidebar.header("Editing panel")
        # writing settings code
        st.sidebar.write("Settings")
        setting_sharp = st.sidebar.slider("Sharpness")
        setting_color = st.sidebar.slider("Color")
        setting_brightness = st.sidebar.slider("Exposure")
        setting_contrast = st.sidebar.slider("Contrast")
        setting_flip_image = st.sidebar.selectbox("Flip Image", options=(
            "select flip direction", "FLIP_TOP_BOTTOM", "FLIP_LEFT_RIGHT"))

        # writing filters code
        st.sidebar.write("Filters")
        filter_black_and_white = st.sidebar.checkbox("Black and white")
        filter_blur = st.sidebar.checkbox("Blur")

        if filter_blur:
            filter_blur_strength = st.sidebar.slider("Select Blur strength")

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

        # implementing sharpness
        sharp = ImageEnhance.Sharpness(img)
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

        # implementing filters
        if filter_black_and_white:
            edited_img = edited_img.convert(mode='L')

        if filter_blur:
            if filter_blur_strength:
                set_blur = filter_blur_strength
                edited_img = edited_img.filter(ImageFilter.GaussianBlur(set_blur))

        # displaying edited image

        if st.button('Colorize'):
            sess, input_name = load_onnx_model(r'F:\Facultate\Dizertatie\Proiect\models\large_model_simplified.onnx')
            image_cv2 = np.array(img)
            h, w = image_cv2.shape[:2]
            tensor_rgb = preprocess_image_for_inference(image_cv2, 512)
            image_cv2 = (image_cv2 / 255.0).astype(np.float32)
            orig_l = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2Lab)[:, :, :1]
            result = run_inference(tensor_rgb, input_name, sess, w, h, orig_l)
            edited_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        st.image(edited_img)

        img_bytes = io.BytesIO()
        edited_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        st.download_button('Download image', data=img_bytes, file_name=f'revived_image_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.png', mime='image/png')


if __name__ == "__main__":
    main()
