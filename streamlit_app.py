import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image


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
@st.dialog("Crop the image to the character")
def crop(img):
    cropped_img = st_cropper(img, realtime_update=True, aspect_ratio=None)
    st.write("Preview")
    st.image(
        cropped_img,
    )
    if st.button("Save Crop"):
        st.session_state.img = cropped_img
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
                    cropped_img = st.session_state.img
                    st.image(
                        cropped_img,
                    )
            else:
                if "img" in st.session_state:
                    st.session_state.pop("img")
                st.error("Please upload an image")

    with col2:
        character = st.text_input(
            "Enter Character",
            max_chars=1,
            disabled="img" not in st.session_state,
            placeholder="F",
        )

    predict_button = st.button(
        "Predict",
        use_container_width=True,
        disabled="img" not in st.session_state or not character,
    )

# Prediction
if predict_button:
    st.divider()
    st.header("Output")

    if "img" in st.session_state:
        st.success(f"Font: ")
    else:
        st.warning("Please upload an image to predict the font")

st.divider()

# About App
st.header("How does the App work?")
st.write(
    "The App uses a **Machine Learning Model** to predict the font used in the input provided by the user. The model is trained on a dataset of fonts and their corresponding images."
)

# Dataset Info
st.header("About the Dataset")
st.write(
    "The dataset used to train the model is the **[Font Dataset](https://archive.ics.uci.edu/dataset/417/character+font+images)**. The dataset contains images from 153 character fonts."
)
with st.expander("Show Dataset Description"):
    st.write(
        "The dataset contains images from 153 character fonts. Each font has 20 images of each character. The dataset contains images of the following characters: **A-Z, a-z, 0-9**."
    )
    # Add sample data from dataset

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
