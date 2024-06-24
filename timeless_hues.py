import io
from datetime import datetime
import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance

from ML_Side.load_onnx_model import *
from Utils.image_processing import colorize_image, colorfulness_score_mapping, compute_colorfulness_score
from Utils.config_files import load_onnx_model_info
from Web_Side.OptionsClass import ImageOptions

CACHE_DIR = Path.cwd().joinpath('cache_files')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CONFIG_PATH = 'Timeless_Hues\ML_Side\model_loading_config.yaml'


def main():
    st.set_page_config(page_title="Timeless Hues")
    st.header("Timeless Hues 🤖+🎨🖌️ ➜ 🖼️")
    st.subheader("Upload an image to get started")
    image = st.file_uploader("Upload an image", type=["png", "jpg"], accept_multiple_files=False, )

    if image:
        if CACHE_DIR.joinpath(f'{Path(image.name).stem}.npy').exists():
            edited_img = Image.fromarray(np.load(CACHE_DIR.joinpath(f'{Path(image.name).stem}.npy')))
        else:
            img = Image.open(image)
            edited_img = img

        options = ImageOptions()

        st.sidebar.header("Editing panel")
        st.sidebar.write("Settings")

        setting_sharp = st.sidebar.slider("Sharpness", min_value=0.01, max_value=10.0, value=1.0)  # done
        setting_color = st.sidebar.slider("Color", min_value=0.01, max_value=10.0, value=1.0)  # done
        setting_exposure = st.sidebar.slider("Exposure", min_value=0.01, max_value=10.0, value=1.0)  # done
        setting_contrast = st.sidebar.slider("Contrast", min_value=0.01, max_value=10.0, value=1.0)  # done

        setting_flip_image = st.sidebar.selectbox("Flip Image", options=("select flip direction", "FLIP_TOP_BOTTOM", "FLIP_LEFT_RIGHT"))
        colorize_button = st.button('Colorize')

        st.sidebar.write("Filters")
        filter_blur = st.sidebar.checkbox("Blur")

        if colorize_button:
            enable_colorize = True
        else:
            enable_colorize = False

        if filter_blur:
            filter_blur_strength = st.sidebar.slider("Select Blur strength", min_value=0.00, max_value=10.0, value=0.0)

        # checking sharpness
        if setting_sharp:
            options.sharpness = setting_sharp

        # checking color
        if setting_color:
            options.color = setting_color

        # checking brightness
        if setting_exposure:
            options.exposure = setting_exposure

        # checking contrast
        if setting_contrast:
            options.contrast = setting_contrast

        # checking setting_flip_image
        flip_direction = setting_flip_image

        # colorizing image:
        with st.spinner('Colorizing...'):
            model_name, model_size = load_onnx_model_info(MODEL_CONFIG_PATH)
            edited_img = colorize_image(model_name, model_size, edited_img, image.name, enable_colorize)
            if enable_colorize:
                logging.info(f'Colorizing was applied on image {image.name}. Local cache {f"{Path(image.name).stem}.npy"} was created or updated!')

        # implementing sharpness
        sharp = ImageEnhance.Sharpness(edited_img)
        # edited_img = sharp.enhance(sharp_value)
        edited_img = sharp.enhance(options.sharpness)

        # implementing color balance
        color = ImageEnhance.Color(edited_img)
        edited_img = color.enhance(options.color)

        # implementing exposure
        brightness = ImageEnhance.Brightness(edited_img)
        edited_img = brightness.enhance(options.exposure)

        # implementing contrast
        contrast = ImageEnhance.Contrast(edited_img)
        edited_img = contrast.enhance(options.contrast)

        # implementing flip direction
        if flip_direction == "FLIP_TOP_BOTTOM":
            edited_img = edited_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif flip_direction == "FLIP_LEFT_RIGHT":
            edited_img = edited_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            pass

        if filter_blur:
            if filter_blur_strength:
                options.blur = filter_blur_strength
                edited_img = edited_img.filter(ImageFilter.GaussianBlur(options.blur))

        st.image(edited_img, use_column_width=True)
        score = round(compute_colorfulness_score(np.array(edited_img)), 4)
        st.text(f'Colorfulness score: {score} ({colorfulness_score_mapping(score)})')

        img_bytes = io.BytesIO()
        edited_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        st.download_button('Download image', data=img_bytes, file_name=f'revived_image_{datetime.now().strftime("%d_%m_%Y_%H:%M")}.png', mime='image/png')
        if st.button('Clear cache'):
            if CACHE_DIR.joinpath(f'{Path(image.name).stem}.npy').exists():
                CACHE_DIR.joinpath(f'{Path(image.name).stem}.npy').unlink(missing_ok=True)
                logging.info(f'Local cache for image {image.name} was deleted!')


if __name__ == "__main__":
    main()
