from sklearn.metrics import accuracy_score
import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Graphology Analysis", layout="wide")

# --- Load Model and Scaler ---
model = joblib.load("models/model.pkl")  # Load single model file
scaler = joblib.load("models/scaler.pkl")  # Load single scaler file

# --- Personality Mapping ---
personality_map = {
    0: {"name": "Introverted & Thoughtful", "description": "🧠 Prefers solitude, enjoys deep thinking, and is highly reflective."},
    1: {"name": "Outgoing & Confident", "description": "🎉 Sociable, enjoys engaging with others, and exudes confidence."},
    2: {"name": "Creative & Expressive", "description": "🎨 Imaginative, highly expressive, and values artistic expression."},
    3: {"name": "Analytical & Detail-Oriented", "description": "📊 Focused on precision, prefers logic over emotions."},
    4: {"name": "Empathetic & Compassionate", "description": "❤️ Emotionally attuned, values deep connections, and is highly empathetic."}
}

# --- Feature Descriptions ---
feature_descriptions = {
    "baseline_angle": {
        "low": "➡️ Slightly inclined writing suggests **calm**, **stability**, and **control**.", 
        "high": "↘️ Highly inclined writing indicates **spontaneity**, **impulsiveness**, or **creativity.**"
    },
    "letter_size": {
        "low": "🔍 Small letters suggest **introversion**, **focus**, and **attention to detail**.",
        "high": "🌟 Large letters indicate **outgoing nature**, **confidence**, and **expressiveness**."
    },
    "line_spacing": {
        "low": "👥 Closely spaced lines suggest **high emotional intensity and impatience**.",
        "high": "⏰ Widely spaced lines indicate **calmness, patience, and a relaxed attitude**."
    },
    "word_spacing": {
        "low": "🔒 Narrow word spacing suggests being **reserved, cautious, and guarded**.",
        "high": "🚀 Wide word spacing indicates **openness, sociability, and independence**."
    },
    "pen_pressure": {
        "low": "🧯 Light pressure shows **sensitivity, empathy, and delicacy**.",
        "high": "⚡ Heavy pressure indicates **determination, passion, and high emotional intensity**."
    }
}

# --- Preprocessing ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    return thresh

# --- Slant Angle Estimation with Multiple Categories ---
def estimate_slant_angle(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        if len(c) >= 5:
            _, _, angle = cv2.fitEllipse(c)
            angles.append(angle)
    avg_angle = np.mean(angles) if angles else 0

    # Categorizing slant angle into six parameters
    if avg_angle < -45:
        slant_category = "More Left"
    elif -45 <= avg_angle < -20:
        slant_category = "Left"
    elif -20 <= avg_angle < -5:
        slant_category = "Slightly Left"
    elif -5 <= avg_angle <= 5:
        slant_category = "Straight"
    elif 5 < avg_angle <= 20:
        slant_category = "Slightly Right"
    elif 20 < avg_angle <= 45:
        slant_category = "Right"
    else:
        slant_category = "More Right"

    return avg_angle, slant_category

# --- Feature Extraction ---
def extract_all_features(image):
    processed_img = preprocess_image(image)
    slant_angle_value, slant_category = estimate_slant_angle(processed_img)

    features = {
        'baseline_angle': np.random.uniform(-15, 15),  # Placeholder
        'letter_size': np.random.uniform(5, 50),  # Placeholder
        'line_spacing': np.random.uniform(5, 30),  # Placeholder
        'word_spacing': np.random.uniform(5, 50),  # Placeholder
        'pen_pressure': np.random.uniform(50, 200),  # Placeholder
        'slant_angle': slant_angle_value  # Actual calculated value
    }

    return features, slant_category

# --- Streamlit UI ---
st.title("📝 Decoding Emotion with Your Handwriting")

uploaded_file = st.file_uploader("📤 Upload a handwriting sample (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="📸 Uploaded Handwriting Sample", use_column_width=True)

    features, slant_category = extract_all_features(image)
    feature_values = np.array([[features['baseline_angle'],
                                features['letter_size'],
                                features['line_spacing'],
                                features['word_spacing'],
                                features['pen_pressure'],
                                features['slant_angle']]])
    scaled_features = scaler.transform(feature_values)
    prediction = model.predict(scaled_features)

    # --- Personality Prediction ---
    pred_class = prediction[0]
    personality = personality_map.get(pred_class, {
        "name": "Unknown",
        "description": "❓ No matching personality found. Please check your input."
    })

    st.success(f"🎯 **Primary Personality:** {personality['name']}")
    st.write(personality['description'])

    # --- Detailed Analysis ---
    st.subheader("✍️ Detailed Handwriting Characteristics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("📏 Baseline Angle", f"{features['baseline_angle']:.1f}°")
        st.metric("🔠 Letter Size", f"{features['letter_size']:.1f} px")
        st.metric("🔡 Word Spacing", f"{features['word_spacing']:.1f} px")

    with col2:
        st.metric("📚 Line Spacing", f"{features['line_spacing']:.1f} px")
        st.metric("💪 Pen Pressure", f"{features['pen_pressure']:.1f}")
        st.metric("🧭 Slant Angle", f"{features['slant_angle']:.1f}° ({slant_category})")

