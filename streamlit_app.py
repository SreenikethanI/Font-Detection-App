import streamlit as st
from streamlit_cropper import st_cropper
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Font detection app",
    layout="wide",
)
st.title("🔠 Font Detection App ig?")
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
    cropped_img, box = st_cropper(
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
    # st.image(cropped_img)
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
    column_config={"Confidence": st.column_config.ProgressColumn(width="medium")},
)

top_font = output[0]["Font"]
st.success(f"Top Font: {top_font}")

# indent section ends here

st.divider()

# Dataset Info
st.header("About the Dataset")
st.write(
    """
    The dataset used to train the model is the **[Font Dataset](https://archive.ics.uci.edu/dataset/417/character+font+images)**. 
    This dataset contains images from 153 different character fonts. Each font includes images of the following characters: **A-Z, a-z, 0-9**.
    The images are grayscale and have been preprocessed to a uniform size. The diversity of fonts in the dataset helps the model learn to 
    distinguish between subtle differences in character shapes and styles, making it robust for font detection tasks.
    """
)
with st.expander("Sample from Dataset"):
    st.write(
        "The dataset contains images from 153 character fonts. The dataset contains images of the following characters: **A-Z, a-z, 0-9**."
    )
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
st.write(
    "The model used in this App is a **Convolutional Neural Network (CNN)**. The CNN is trained on the Font Dataset to predict the font used in an input provided by the user."
)

st.divider()

st.write(
    """
## About the Developers

         \n- **Eman**
         \n- **Joe**
         \n- **eeni**
         \n- **Yusi**\n

## References

         \n- **[Font Dataset](https://archive.ics.uci.edu/dataset/417/character+font+images)**
         \n- **[Pillow](https://pillow.readthedocs.io/en/stable/)**
         \n- **[idk](idk sm)**
    """
)
