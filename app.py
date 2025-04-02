import time
import streamlit as st
from PIL import Image
from utils import predict_image  # Import prediction logic

# Streamlit App Configuration
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

def prediction_page():
    st.sidebar.title("🌱 Navigation")
    page = st.sidebar.radio("Go to", ["🌿 Predict Disease", "📚 About", "❓ Help & Resources"])

    if page == "🌿 Predict Disease":
        st.title("🌿 Plant Disease Classification")
        st.write("Upload plant leaf images to detect diseases.")

        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png","webp","bmp","jpeg","svg"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

                with col2:
                    with st.spinner("Analyzing..."):
                        time.sleep(2)
                        img = Image.open(uploaded_file)
                        predicted_label, predicted_prob, disease_solution, disease_link = predict_image(img)
                    
                    st.subheader("Prediction Result")
                    st.markdown(f"### 🏷️ {predicted_label}")
                    st.progress(predicted_prob)
                    st.markdown(f"**Confidence:** {predicted_prob * 100:.2f}%")
                    st.subheader("Treatment Solution")
                    st.write(disease_solution)
                    if disease_link:
                        st.markdown(f"[Click here for more information]({disease_link})")

    elif page == "📚 About":
        st.title("📚 About the Project")
        st.write("""
            This app was developed to help farmers and plant enthusiasts identify diseases affecting their plants.
            The model was trained on a dataset of plant diseases to assist in early detection.
        """)
        st.write("**Developed by:** Ayyanar")

    elif page == "❓ Help & Resources":
        st.title("❓ Help & Resources")
        st.write("Here are some common treatments for detected diseases:")
        st.write("- **Apple Scab**: Use fungicides and prune infected leaves.")
        st.write("- **Tomato Mosaic Virus**: Remove infected plants to prevent spread.")
        st.write("- **Powdery Mildew**: Apply fungicides and improve air circulation.")

if __name__ == "__main__":
    prediction_page()
