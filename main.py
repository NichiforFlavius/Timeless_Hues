# importing modules
import io
from datetime import datetime

import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance

from OptionsClass import ImageOptions

# page configurations
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
    st.image(edited_img, width=400)

    img_bytes = io.BytesIO()
    edited_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    st.download_button('Download image', data=img_bytes, file_name=f'revived_image_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.png', mime='image/png')

    st.write(">To download edited image right click and `click save image as`.")
