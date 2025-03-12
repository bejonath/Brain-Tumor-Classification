import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import io

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    /* Hide empty containers */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Customize the file uploader */
    .stFileUploader > div:first-child {
        background-color: #1E1E1E;
        padding-top: 0rem;
        border: none;
        box-shadow: none;
    }
    .stFileUploader > div > div > div {
        background-color: #2D3748;
        border-radius: 0.5rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #4299E1;
    }
    .subtitle {
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
        color: #A0AEC0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #2D3748;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        color: white;
    }
    .confidence {
        font-size: 1.3rem;
        margin-top: 0.5rem;
        color: white;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #A0AEC0;
        font-size: 0.8rem;
    }
    .upload-section {
        background-color: #2D3748;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 0.25rem;
    }
    .color-legend {
        margin-top: 1rem;
        color: white;
    }
    .color-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        color: white;
    }
    /* Remove extra whitespace */
    div[data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0 !important;
    }
    .stHeader {
        display: none;
    }
    /* Remove white boxes */
    [data-testid="stAppViewContainer"] > div:first-child {
        background-color: #1E1E1E !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
    }
    .stTextInput > div:first-child {
        background-color: transparent;
    }
    .stFileUploader label {
        display: none;
    }
    .css-184tjsw p {
        font-size: 1rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(r"D:\Personal\Projects\braintumor1\CNN\cnn_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

def add_colored_border(img, tumor_detected):
    img_with_border = img.copy()
    border_color = (255, 0, 0) if tumor_detected else (0, 255, 0)  # Red if tumor, Green if no tumor
    
    if img_with_border.mode != 'RGB':
        img_with_border = img_with_border.convert('RGB')
    
    border_width = 15
    bordered_img = ImageOps.expand(img_with_border, border=border_width, fill=border_color)
    
    return bordered_img

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

try:
    model = load_model()
    
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.markdown('<div class="title">Brain Tumor</div>', unsafe_allow_html=True)
        st.markdown('<div class="title">Classification</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Upload a brain MRI scan to detect and classify tumors</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload an MRI scan image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            st.session_state.uploaded_image = Image.open(uploaded_file)
            st.session_state.prediction_made = False  
        
        if st.button("Predict", disabled=st.session_state.uploaded_image is None, use_container_width=True):
            if st.session_state.uploaded_image is not None:
                with st.spinner("Analyzing the image..."):
                    processed_img = preprocess_image(st.session_state.uploaded_image)
                    
                    prediction = model.predict(processed_img)
                    predicted_class_index = np.argmax(prediction, axis=1)[0]
                    predicted_class = classes[predicted_class_index]
                    confidence = float(prediction[0][predicted_class_index]) * 100
                    
                    tumor_detected = predicted_class != 'notumor'
                    
                    bordered_img = add_colored_border(st.session_state.uploaded_image, tumor_detected)
                    
                    st.session_state.prediction_result = {
                        'image': bordered_img,
                        'class': predicted_class,
                        'confidence': confidence,
                        'tumor_detected': tumor_detected
                    }
                    
                    st.session_state.prediction_made = True
        
    
    # Right column: Display image and prediction results
    with right_col:
        if st.session_state.uploaded_image is not None:
            if st.session_state.prediction_made and st.session_state.prediction_result is not None:
                st.image(st.session_state.prediction_result['image'], width=400)
                
                st.markdown(f'<div class="prediction-result">Prediction: {st.session_state.prediction_result["class"].capitalize()}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">Confidence: {st.session_state.prediction_result["confidence"]:.2f}%</div>', unsafe_allow_html=True)
                
                status_color = "🔴" if st.session_state.prediction_result["tumor_detected"] else "🟢"
                status_text = "Tumor Detected" if st.session_state.prediction_result["tumor_detected"] else "No Tumor Detected"
                st.markdown(f"<div style='font-size:1.3rem; margin-top:0.5rem; color:white;'>Status: {status_color} {status_text}</div>", unsafe_allow_html=True)
                
            else:
                st.image(st.session_state.uploaded_image, width=400)
                st.markdown("<div style='text-align:center; margin-top:1rem; color:white;'>Click 'Predict' to analyze this image</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="footer">This application is for educational purposes only and should not be used for medical diagnosis.</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred: {e}")