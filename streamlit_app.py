# MARK: Import Statements
import streamlit as st
from streamlit_cropper import st_cropper
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from model.model import SimbleModel

# MARK: Creating and loading model
MODEL_PATH = "model/e29 b00324 - l 0.94693 (complete).pth"
FONTNAMES_PATH = "model/fontnames all.txt"

if "model" not in st.session_state:
    st.session_state["model"] = SimbleModel(MODEL_PATH, FONTNAMES_PATH)

# MARK: Content/Tutorial
st.set_page_config(
    page_title="Font detection app",
    layout="wide",
)
st.title("🔠 Machine Learning Based Font Detection App")
st.write(
    """
This is a demo of the assignment in the Foundations of Data Science Course (CS F320). The app allows users to upload images of text, and it will predict the font used in the image. The model has been trained on a diverse set of fonts to ensure accurate recognition.
    """
)

with st.expander("**How to use the App?**", icon="❓"):
    st.write(
    """
    1. **Upload an Image**:
    Click the **"Upload Image"** button and select an image file containing the text or character you want to analyze. Supported formats: **JPG, PNG, JPEG**.

    2. **Crop the Image**:
    Once the image is uploaded, a cropping dialog will appear. Adjust the crop area to isolate the character you want to detect. When done, click **"Save Crop"** to store the cropped character.

    3. **Enter the Character**:
    In the text input field, type the character you cropped. **Make sure it matches exactly** (case-sensitive) to ensure accurate predictions.

    4. **Predict the Font**:
    Click the **"Predict"** button to process your input through the model. The app will display the predicted font in the **Output** section.
    """
    )

st.divider()

# MARK: Input Section
st.header("✒️ Inputs")


# fn for crop dialog
# MARK: Crop Dialog
@st.dialog("🖼️ Crop the image to the character", width="large")
def crop(img: Image.Image):

    st.write(
        """
             Adjust the crop area to isolate the character you want to detect.
             Adjust the Sliders and Invert image to make the character more visible.
             """
    )
    SCALE = st.slider("Zoom", 1, 5, 2)
    CONTRAST = st.slider("Contrast", 0.5, 2.0, 1.0)
    BRIGHTNESS = st.slider("Brightness", 0.0, 2.0, 1.0)

    box = st_cropper(
        img.resize((img.width * SCALE, img.height * SCALE)),
        realtime_update=True,
        aspect_ratio=None,
        should_resize_image=True,
        return_type="box",
    )

    # left, upper, right, lower
    final_image = img.crop(
        (
            (box["left"]) / SCALE,
            (box["top"]) / SCALE,
            (box["left"] + box["width"]) / SCALE,
            (box["top"] + box["height"]) / SCALE,
        )
    )

    final_image = final_image.convert("L")
    enhancer = ImageEnhance.Contrast(final_image)
    final_image = enhancer.enhance(CONTRAST)
    enhancer = ImageEnhance.Brightness(final_image)
    final_image = enhancer.enhance(BRIGHTNESS)

    inv = st.checkbox("Invert Image", False)
    final_image = ImageOps.invert(final_image) if inv else final_image

    st.write("Preview")
    st.image(final_image)
    if st.button("Save Crop"):
        st.session_state.img = final_image
        st.rerun()


# MARK: Upload and Character Input
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            font_image = st.file_uploader(
                "📤 Upload Image",
                type=["jpg", "png", "jpeg"],
                accept_multiple_files=False,
            )

            if font_image:
                if "img" not in st.session_state:
                    st.session_state.img = None
                    img = Image.open(font_image)
                    crop(img)
                else:
                    st.image(
                        st.session_state.img,
                    )
                    if st.button("Change Image"):
                        st.session_state.pop("img")
                        img = Image.open(font_image)
                        crop(img)
            else:
                if "img" in st.session_state:
                    st.session_state.pop("img")
                st.error("⚠️ Please upload an image")

    with col2:
        character = st.text_input(
            "🅱️ Enter Character",
            max_chars=1,
            placeholder="B",
        )

    predict_button = st.button(
        "Predict",
        use_container_width=True,
        disabled="img" not in st.session_state or not character,
    )

# MARK: Prediction
if predict_button:
    if "model" not in st.session_state:
        st.error("⚠️ The model is not loaded yet, please try again.")

    else:
        st.divider()

        # MARK: Output Section
        st.header("📊 Output")

        output = [
            {"Font": font_name, "Confidence": score}
            for font_name, score in st.session_state.model.predict(
                st.session_state.img, ord(character)
            )
        ]

        prediction = pd.DataFrame(output)

        st.dataframe(
            prediction,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(width="large")
            },
        )

        top_font = output[0]["Font"]
        st.success(f"Top font: {top_font}")

st.divider()

# MARK: Dataset Info
st.header("📈 About the Dataset")
st.write(
    """
The dataset used to train the model is synthetically generated with the help of the **[Pillow](https://pillow.readthedocs.io/en/stable/)** library. It is created using 173 commonly available fonts found on most modern Windows devices.

For each font, images are generated for a wide range of characters, including:
- **Uppercase letters**: `A-Z`
- **Lowercase letters**: `a-z`
- **Digits**: `0-9`

Each image is preprocessed to a consistent size of **32x32 pixels** and stored in a dataset for training.

The goal of using synthetic data is to create a highly diverse dataset that helps the model generalize well across different fonts and character styles. By training on a wide variety of fonts, the model learns to recognize subtle differences in character shapes and styles, making it robust and reliable for font detection tasks.
"""
)

with st.expander("Fonts Used in Dataset", icon="❗"):
    df = pd.read_csv("frontend_demo.csv")
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Font Name": st.column_config.Column(width="small"),
            "Sample Text": st.column_config.ImageColumn(
                label=None, width="large", help=None
            ),
        },
        use_container_width=True,
    )

# MARK: Model Info
st.header("🤖 About the Model")
st.write(
    """
The model used in this app is a **Convolutional Neural Network (CNN)** trained on the font dataset. CNNs are a class of deep learning models that are particularly effective for image classification tasks.  
More details on the model can be found on @SreenikethanI's Repository [here]().
    """
)

with st.expander("Confusion Matrix", icon="⁉️"):
    st.image(
        "assets/Confusion Matrix (P).png",
        use_container_width=True,
        caption='Confusion Matrix of the Character "P"',
    )

st.divider()

# MARK: Team and References
st.write(
    """
## 💻 About the Developers

- **Yusra Hakim - 2022A7PS0004U**
- **Joseph Cijo - 2022A7PS0019U**
- **Sreenikethan Iyer - 2022A7PS0034U**
- **Mohammed Emaan - 2022A7PS0036U**

## 📚 References

- **[Font Dataset](https://archive.ics.uci.edu/dataset/417/character+font+images)**
- **[Pillow](https://pillow.readthedocs.io/en/stable/)**
- **Z. Wang, J. Yang, H. Jin, et. al., "DeepFont: Identify Your Font from An Image" [Available (DOI)](https://doi.org/10.48550/arXiv.1507.03196)**
    """
)
