import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Set page config with wider layout
st.set_page_config(
    page_title="Digit Drawing Recognition Web App",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .header-logo {
        margin-right: 1rem;
    }
    .header-title {
        color: #2c3e50;
        margin: 0;
        font-size: 1.5rem;
    }
    .canvas-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .results-box {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: #4a6fa5;
        color: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        background-color: #e0e0e0;
        margin: 1rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
    }
    .digit-image {
        border: 1px solid #eee;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .graph-container {
        margin-top: 1rem;
    }
    /* Make everything more compact */
    .stButton>button {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_keras_model():
    try:
        model = load_model('Final_Digit_Classify_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Load model
model = load_keras_model()

# Header section with logo left of title
st.markdown("""
<div class="header-container">
    <div class="header-logo">
        <img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" width="60">
    </div>
    <div>
        <h1 class="header-title">Digit Recognition</h1>
        <p style="margin:0;">Draw any digit from 0-9</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1,1], gap="medium")

with col1:
    st.markdown("<div class='canvas-container'>", unsafe_allow_html=True)
    canvas = st_canvas(
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=250,
        width=250,
        drawing_mode="freedraw",
        key="canvas"
    )
    if st.button("Clear Canvas", use_container_width=True):
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='results-box'>", unsafe_allow_html=True)
    
    if canvas.image_data is not None and np.any(canvas.image_data != 0):
        with st.spinner('Analyzing...'):
            try:
                # Process image
                img = Image.fromarray(canvas.image_data.astype('uint8'))
                img = img.resize((28, 28), Image.LANCZOS)
                img = ImageOps.grayscale(img)
                img_array = np.array(img).astype('float32') / 255.0
                
                # Prepare model input
                model_input = img_array.reshape(1, 784) if model.input_shape[1] == 784 else img_array.reshape(1, 28, 28, 1)
                
                # Get prediction
                prediction = model.predict(model_input)
                pred_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Display results
                st.markdown("**Results**")
                
                # Prediction card
                st.markdown(f"""
                <div class='prediction-card'>
                    <h3 style='margin:0;'>Predicted: {pred_digit}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                st.write(f"Confidence: {confidence:.1%}")
                st.markdown(f"""
                <div class='confidence-bar'>
                    <div class='confidence-fill' style='width: {confidence*100}%'></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Processed image
                st.image(img_array, 
                        caption="What AI sees", 
                        width=120)
                
                # Automatic graph display
                st.markdown("<div class='graph-container'>", unsafe_allow_html=True)
                st.bar_chart(prediction[0], height=200)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("Draw a digit to see predictions")
        st.image("https://cdn-icons-png.flaticon.com/512/3462/3462466.png", width=120)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 0.5rem; font-size: 0.8rem;">
    <p style="margin:0;">Powered by Deep Learning</p>
    <h4 style="margin:0;"> Develop by Muhammad Uzair</h4>
</div>
""", unsafe_allow_html=True)