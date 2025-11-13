import streamlit as st

st.set_page_config(page_title="Mentra AI", layout="wide")

# --- Remove Streamlit default UI ---
st.markdown("""
<style>
/* Remove padding and default Streamlit UI */
.block-container, .main, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    padding: 0 !important;
    margin: 0 !important;
    visibility: hidden !important;
    height: 0 !important;
}

/* Full screen gradient background */
.stApp {
    background: radial-gradient(circle at bottom right, #1b1b65 0%, #0a043c 40%, #010326 100%);
    background-size: 400% 400%;
    animation: gradientShift 18s ease-in-out infinite;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: white;
    height: 100vh;
    overflow: hidden;
}

/* Avatar image circular + floating */
.avatar {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    margin-bottom: 25px;
    box-shadow: 0 0 30px rgba(80, 90, 255, 0.6);
    animation: floatAvatar 6s ease-in-out infinite;
}

/* Headline */
h1 {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 15px;
    text-shadow: 0 0 40px rgba(90, 100, 255, 0.5);
    animation: floatHero 7s ease-in-out infinite;
}

/* Paragraph */
p {
    font-size: 1.2rem;
    opacity: 0.9;
    margin-bottom: 30px;
    animation: floatText 7s ease-in-out infinite;
}

/* Buttons container */
.buttons {
    display: flex;
    justify-content: center;
    gap: 1rem;
    animation: floatButtons 6s ease-in-out infinite;
}

/* Button styles */
.btn-primary {
    padding: 14px 30px;
    border-radius: 12px;
    font-weight: 600;
    text-decoration: none;
    background: linear-gradient(90deg, #5b6bff, #3a45ff);
    color: white;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
}
.btn-primary:hover {
    transform: translateY(-5px) scale(1.05);
    box-shadow: 0 0 35px rgba(80, 90, 255, 0.8);
}

.btn-secondary {
    padding: 14px 30px;
    border-radius: 12px;
    font-weight: 600;
    text-decoration: none;
    color: white;
    border: 1.5px solid rgba(255, 255, 255, 0.6);
    background: transparent;
    cursor: pointer;
    transition: all 0.3s ease;
}
.btn-secondary:hover {
    background: rgba(255,255,255,0.1);
    transform: translateY(-5px) scale(1.05);
    box-shadow: 0 0 25px rgba(255,255,255,0.3);
}

/* Animations */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes floatAvatar {
    0%, 100% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-12px) scale(1.05); }
}
@keyframes floatHero {
    0%, 100% { transform: translateY(0px) scale(1); text-shadow: 0 0 40px rgba(90,100,255,0.4);}
    50% { transform: translateY(-18px) scale(1.03) rotateX(2deg); text-shadow: 0 0 80px rgba(100,110,255,0.8);}
}
@keyframes floatText {
    0%, 100% { transform: translateY(0px); opacity: 0.9; }
    50% { transform: translateY(-10px); opacity: 1; }
}
@keyframes floatButtons {
    0%,100%{transform:translateY(0px) scale(1);}
    50%{transform:translateY(-12px) scale(1.02);}
}
</style>
""", unsafe_allow_html=True)

# --- Page content ---
st.markdown("""
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
    <img src="moot.jpg" class="avatar" />
    <h1>Mentra AI</h1>
    <p>Your intelligent assistant for predicting academic success and preventing dropouts.</p>
    <div class="buttons">
        <a href="dashboard.html" class="btn-primary" target="_blank">Try Mentra AI</a>
        <a href="learnmore.html" class="btn-secondary">Learn More</a>
    </div>
</div>
""", unsafe_allow_html=True)
