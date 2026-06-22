"""
TB Detection Streamlit App
Integrates frontend design with ResNet50 backend model
"""
import streamlit as st
import sys
from pathlib import Path
import time
from PIL import Image
import io

# Add backend to path
backend_path = Path(__file__).parent.parent / "BACKEND_FINAL"
sys.path.insert(0, str(backend_path))

from inference import XRayPredictor

# Page config
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching your frontend design
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .success-badge {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    .warning-badge {
        background: #f59e0b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    .danger-badge {
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize predictor
@st.cache_resource
def load_predictor():
    """Load model once and cache"""
    return XRayPredictor()

# Header
st.markdown("<h1>🫁 TB Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-assisted chest X-ray triage for faster, consistent screening workflows</p>", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Chest X-ray")
    st.write("Select a frontal chest radiograph image to run automated TB screening.")
    
    uploaded_file = st.file_uploader(
        "Diagnostic Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a chest X-ray image (JPG, PNG, or BMP format)"
    )
    
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)
    
    analyze_button = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("📊 Analysis Results")
    
    if analyze_button and uploaded_file:
        # Save uploaded file temporarily
        temp_path = Path("temp_upload.jpg")
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Show loading
        with st.spinner("🔄 Analyzing image..."):
            try:
                # Load predictor
                predictor = load_predictor()
                
                # Get prediction
                result = predictor.predict(str(temp_path), return_timing=True)
                
                # Display results
                disease = result['disease']
                confidence = result['confidence']
                
                # Disease badge
                if disease == "Normal":
                    badge_class = "success-badge"
                    emoji = "✅"
                elif disease == "Pneumonia":
                    badge_class = "warning-badge"
                    emoji = "⚠️"
                else:  # Tuberculosis
                    badge_class = "danger-badge"
                    emoji = "🔴"
                
                st.markdown(f"<div class='{badge_class}'>{emoji} {disease}</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Confidence</div>
                        <div class='metric-value'>{confidence*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    inference_time = result.get('timing', {}).get('total_time', 0)
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Processing Time</div>
                        <div class='metric-value'>{inference_time:.2f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed results
                st.markdown("---")
                st.subheader("📋 Detailed Analysis")
                
                # All class probabilities
                if 'all_probabilities' in result:
                    st.write("**Class Probabilities:**")
                    probs = result['all_probabilities']
                    for class_name, prob in probs.items():
                        st.progress(prob, text=f"{class_name}: {prob*100:.2f}%")
                
                # Clinical recommendation
                st.markdown("---")
                st.subheader("💡 Clinical Recommendation")
                
                if disease == "Normal":
                    st.success("✅ No significant abnormalities detected. Routine follow-up recommended.")
                elif disease == "Pneumonia":
                    st.warning("⚠️ Pneumonia suspected. Please consult a doctor for clinical evaluation and appropriate treatment.")
                else:  # Tuberculosis
                    st.error("🔴 **Tuberculosis suspected.** Please consult a doctor immediately for:\n\n- Clinical evaluation\n- Sputum testing (AFB smear/culture)\n- Chest CT scan if needed\n- Appropriate TB treatment protocol")
                
                # Disclaimer
                st.markdown("---")
                st.caption("⚠️ This is a decision support tool. All results must be reviewed by qualified medical professionals. Not approved for clinical diagnosis without expert validation.")
                
            except Exception as e:
                st.error(f"❌ Error analyzing image: {str(e)}")
            
            finally:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
    
    elif analyze_button:
        st.warning("⚠️ Please upload an image first.")
    else:
        st.info("👆 Upload a chest X-ray image and click 'Analyze Image' to begin.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: white;'>Capstone Project | Powered by ResNet50 Deep Learning Model</p>", unsafe_allow_html=True)
