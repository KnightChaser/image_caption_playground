import streamlit as st
from utils import load_image
from caption_models import CaptionModel

# Initialize model (cache it so not loading each time)
@st.cache_resource
def load_caption_model():
    return CaptionModel()

model = load_caption_model()

st.set_page_config(page_title="Image Captioning Demo", layout="wide")
st.title("Image → Description ⚡")

uploaded_file = st.file_uploader("Upload an image (jpg/jpeg/png)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="🔍 Uploaded Image", use_column_width=True)

    if st.button("Generate Description"):
        with st.spinner("Generating caption…"):
            caption = model.predict(image)
        st.success("✅ Caption generated!")
        st.markdown(f"**Description:** {caption}")

