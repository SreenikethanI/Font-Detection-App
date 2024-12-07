import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image


@st.dialog("Crop the image to the character")
def crop(img):
    cropped_img = st_cropper(img, realtime_update=True, aspect_ratio=None)
    st.write("Preview")
    st.image(
        cropped_img,
        caption=f"Character: {character}",
    )
    if st.button("Save Crop"):
        st.session_state.img = cropped_img
        st.rerun()


st.title("🔠 Font Detection App ig?")
st.write(
    "This is a demo of the assignment in the **Foundations of Data Science Course**"
)

st.divider()

st.header("What is This Project about?")
st.write(
    "This project is about predicting the font used in an input provided by the user. The user can input any text and the model will predict the font used in the input."
)

st.header("How does the App work?")
st.write(
    "The App uses a **Machine Learning Model** to predict the font used in the input provided by the user. The model is trained on a dataset of fonts and their corresponding images."
)

st.header("How to use the App?")
st.write(
    "To use the App, simply input the character in the text box provided and upload a screenshot of the same character and click on the **Predict** button. The model will then predict the font used in the input."
)

st.header("About the Dataset")
st.write(
    "The dataset used to train the model is the **[Font Dataset](https://archive.ics.uci.edu/dataset/417/character+font+images)**. The dataset contains images from 153 character fonts."
)

st.header("About the Model")
st.write(
    "The model used in this App is a **Convolutional Neural Network (CNN)**. The CNN is trained on the Font Dataset to predict the font used in an input provided by the user."
)

st.header("About the Developers")
st.write(
    "This App is developed by **Team 1** of the **Foundations of Data Science Course**. The team members are:"
)
st.write("- **Eman**\n" "- **Joe**\n" "- **Sreeni**\n" "- **Yusra**\n")

st.divider()

st.header("Inputs")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        character = st.text_input("Enter Character", max_chars=1)

    with col2:
        with st.container():

            font_image = st.file_uploader(
                "Upload Image",
                type=["jpg", "png", "jpeg"],
                accept_multiple_files=False,
            )

            if font_image:
                if "img" not in st.session_state:
                    img = Image.open(font_image)
                    crop(img)
                else:
                    cropped_img = st.session_state.img
                    st.image(
                        cropped_img,
                        caption=f"Character: {character}",
                    )
            else:
                st.session_state.pop("img")
                st.warning("Please upload an image")

    predict_button = st.button("Predict", use_container_width=True)
