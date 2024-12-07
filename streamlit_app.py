import streamlit as st

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
    "To use the App, simply input any text in the text box provided and click on the **Predict** button. The model will then predict the font used in the input."
)

st.header("About the Dataset")
st.write(
    "The dataset used to train the model is the **Font Dataset**. The dataset contains images of fonts and their corresponding labels."
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
col1, col2 = st.columns(2)
with col1:
    character = st.text_input("Enter Character", max_chars=1)

with col2:
    with st.container():
        font_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if font_image is not None:
            st.image(
                font_image, caption=f"Character: {character}", use_container_width=True
            )

predict_button = st.button("Predict")
