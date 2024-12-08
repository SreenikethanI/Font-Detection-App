import streamlit as st
from streamlit_cropper import st_cropper
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Font detection app",
    layout="wide",
)
st.title("🔠 Machine Learning Based Font Detection App")
st.write(
    "This is a demo of the assignment in the **Foundations of Data Science Course**"
)

st.divider()

# Tutorial
st.header("How to use the App?")
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

# Input Section
st.header("Inputs")


# fn for crop dialog
@st.dialog("Crop the image to the character", width="large")
def crop(img: Image.Image):
    SCALE = st.slider("Zoom", 1, 5, 2)
    box = st_cropper(
        img.resize((img.width * SCALE, img.height * SCALE)),
        realtime_update=True,
        aspect_ratio=None,
        should_resize_image=True,
        return_type="both",
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

    st.write("Preview")
    st.image(final_image)
    if st.button("Save Crop"):
        st.session_state.img = final_image
        st.rerun()


with st.container():
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            font_image = st.file_uploader(
                "Upload Image",
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
                st.error("Please upload an image")

    with col2:
        character = st.text_input(
            "Enter Character",
            max_chars=1,
            placeholder="F",
        )

    predict_button = st.button(
        "Predict",
        use_container_width=True,
        disabled="img" not in st.session_state or not character,
    )

# Prediction
if predict_button:
    print("Predicting...")

# When final prediction is ready, display the output

# indent section starts here)
st.divider()
st.header("Output")

# Mock prediction logic for demonstration purposes
output = [
    {"Font": "Arial", "Confidence": 0.95},
    {"Font": "Helvetica", "Confidence": 0.90},
    {"Font": "Times New Roman", "Confidence": 0.85},
    {"Font": "Courier New", "Confidence": 0.80},
    {"Font": "Verdana", "Confidence": 0.75},
]

prediction = pd.DataFrame(output)

st.dataframe(
    prediction,
    use_container_width=True,
    hide_index=True,
    column_config={"Confidence": st.column_config.ProgressColumn(width="large")},
)

top_font = output[0]["Font"]
st.success(f"Top Font: {top_font}")

# indent section ends here

st.divider()

# Dataset Info
st.header("About the Dataset")
st.write(
    """
The dataset used to train the model is synthetically generated with the help of the **[Pillow](https://pillow.readthedocs.io/en/stable/)** library. It is created using 173 commonly available fonts found on most modern Windows devices.  

For each font, images are generated for a wide range of characters, including:  
- **Uppercase letters**: `A-Z`  
- **Lowercase letters**: `a-z`  
- **Digits**: `0-9`  
- **Symbols**: All common characters on a QWERTY keyboard (e.g., `@`, `#`, `$`, `%`, `&`, etc.)  

Each image is preprocessed to a consistent size of **32x32 pixels** and stored in a dataset for training.  

The goal of using synthetic data is to create a highly diverse dataset that helps the model generalize well across different fonts and character styles. By training on a wide variety of fonts, the model learns to recognize subtle differences in character shapes and styles, making it robust and reliable for font detection tasks.    """
)
with st.expander("Sample from Dataset"):
    # Add sample data from dataset
    df = pd.DataFrame(
        [
            {"command": "st.selectbox", "rating": 4, "is_widget": True},
            {"command": "st.balloons", "rating": 5, "is_widget": False},
            {"command": "st.time_input", "rating": 3, "is_widget": True},
        ]
    )

    st.dataframe(df, use_container_width=True)

# Model Info
st.header("About the Model")
st.write("""
         ### About the ResNet-18 Model  

**ResNet-18** is a deep learning model from the ResNet (Residual Network) family, designed to solve the challenges of training very deep neural networks, particularly the problem of vanishing gradients. It achieves this through the use of **residual blocks**, which introduce *skip connections* that allow the model to learn identity mappings and preserve information across layers.  

With **18 layers**, ResNet-18 strikes a balance between performance and computational efficiency. It is lightweight compared to deeper ResNet variants and is widely used in applications like:  
- **Image Classification**: Categorizing images into predefined labels.  
- **Feature Extraction**: Generating high-quality feature representations for other tasks.  
- **Vision Applications**: Serving as a backbone for object detection and segmentation models.  

ResNet-18's design ensures that it is robust, versatile, and efficient, making it an ideal choice for projects requiring reliable performance with limited computational resources.
         """)

st.divider()

st.write(
    """
## About the Developers

- **Eman**  
- **Joe**  
- **eeni**  
- **Yusi**  

## References

- **[Font Dataset](https://archive.ics.uci.edu/dataset/417/character+font+images)**  
- **[Pillow](https://pillow.readthedocs.io/en/stable/)**  
- **[idk](idk sm)**  
    """
)
