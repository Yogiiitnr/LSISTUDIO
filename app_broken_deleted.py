# coding: utf-8
"""
LSI Acoustic Studio PRO - Premium Interactive DSP Platform
World-class signal processing with famous venues and professional filters
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import butter, cheby1, cheby2, ellip, sosfilt, sosfreqz
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LSI Acoustic Studio PRO",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ENHANCED PREMIUM STYLING - FIXES ALL UI ISSUES
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Root Background */
    html, body {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2a2540 100%);
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2a2540 100%);
        padding: 2rem !important;
    }
    
    /* Premium Header */
    .header-premium {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 30px 60px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255,255,255,0.3);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .header-premium h1 {
        margin: 0;
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -2px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        line-height: 1.1;
    }
    
    .header-premium .subtitle {
        font-size: 1.2rem;
        opacity: 0.98;
        font-weight: 400;
        margin-top: 1rem;
        letter-spacing: 0.3px;
        line-height: 1.5;
    }
    
    /* Section Headers */
    .section-header {
        color: #667eea;
        font-size: 2rem;
        font-weight: 900;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        letter-spacing: -0.5px;
    }
    
    /* BUTTON FIX - MAXIMUM VISIBILITY */
    .stButton {
        display: flex !important;
        width: 100% !important;
    }
    
    .stButton button {
        width: 100% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1.2rem 2rem !important;
        border-radius: 14px !important;
        font-weight: 800 !important;
        letter-spacing: 0.8px !important;
        font-size: 1.05rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        opacity: 1 !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s !important;
    }
    
    .stButton button:hover {
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%) !important;
    }
    
    .stButton button:hover:before {
        left: 100% !important;
    }
    
    .stButton button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
        border: 1.5px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.95;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        margin-top: 1rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    
    .metric-card.accent1 { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .metric-card.accent2 { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .metric-card.accent3 { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .metric-card.accent4 { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent !important;
        border-bottom: 2px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        background: linear-gradient(135deg, #1a1f3a 0%, #16213e 100%) !important;
        font-weight: 700 !important;
        color: #d0d0e0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 2px solid #667eea !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Text Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #667eea;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    h2 {
        border-bottom: 3px solid #667eea;
        padding-bottom: 1rem;
    }
    
    p {
        color: #d0d0e0;
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-left: 5px solid #667eea;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(102, 126, 234, 0.08);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #1a1f3a 0%, #16213e 100%) !important;
    }
    
    /* Input Focus */
    .stSlider, .stSelectbox {
        margin-bottom: 1.5rem;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(67, 233, 123, 0.15) 0%, rgba(56, 249, 215, 0.15) 100%);
        border-left: 5px solid #43e97b;
        border-radius: 8px;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(245, 87, 108, 0.15) 0%, rgba(245, 87, 108, 0.15) 100%);
        border-left: 5px solid #f5576c;
        border-radius: 8px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.15) 0%, rgba(255, 193, 7, 0.15) 100%);
        border-left: 5px solid #ffc107;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ACOUSTIC SPACES DATABASE
# ============================================================================

ACOUSTIC_SPACES = {
    "Taj Mahal (India)": {"decay": 5.5, "reflections": 25, "spacing": 0.045, "emoji": "🕌", "desc": "Warm marble mausoleum"},
    "Sydney Opera House": {"decay": 3.5, "reflections": 18, "spacing": 0.035, "emoji": "🎭", "desc": "Modern concert venue"},
    "Pantheon (Rome)": {"decay": 8.0, "reflections": 35, "spacing": 0.055, "emoji": "🏛️", "desc": "Dome with long reverb"},
    "Grand Central Terminal": {"decay": 4.2, "reflections": 22, "spacing": 0.042, "emoji": "🚂", "desc": "Natural echo chambers"},
    "Ancient Pagoda (Japan)": {"decay": 2.8, "reflections": 15, "spacing": 0.038, "emoji": "🏯", "desc": "Intimate wooden acoustics"},
    "Church of Santo Domingo": {"decay": 6.5, "reflections": 30, "spacing": 0.048, "emoji": "⛪", "desc": "Rich cathedral space"},
    "Sagano Bamboo Grove": {"decay": 1.5, "reflections": 8, "spacing": 0.025, "emoji": "🎋", "desc": "Natural acoustic filter"},
    "Colosseum (Rome)": {"decay": 7.2, "reflections": 40, "spacing": 0.052, "emoji": "🏛️", "desc": "Ancient amphitheater"},
    "Concert Hall": {"decay": 2.2, "reflections": 12, "spacing": 0.032, "emoji": "🎼", "desc": "Optimized for music"},
    "Underwater Cave (Hawaii)": {"decay": 3.8, "reflections": 20, "spacing": 0.044, "emoji": "🌊", "desc": "Water-dampened unique"},
    "Grand Canyon Echo": {"decay": 9.0, "reflections": 45, "spacing": 0.06, "emoji": "🏜️", "desc": "Massive canyon reverb"},
    "Abbey Road Studio": {"decay": 1.8, "reflections": 10, "spacing": 0.028, "emoji": "🎙️", "desc": "Professional recording"},
}

MUSIC_FILTERS = {
    "Vocal Enhancer": {"type": "bandpass", "freq_low": 500, "freq_high": 4000, "desc": "Boost vocal presence"},
    "Bass Booster": {"type": "lowpass", "freq": 200, "desc": "Enhance deep bass"},
    "Treble Shimmer": {"type": "highpass", "freq": 5000, "desc": "Bright airy highs"},
    "Lo-Fi Hip Hop": {"type": "bandpass", "freq_low": 100, "freq_high": 6000, "desc": "Warm retro sound"},
    "Presence Peak": {"type": "bandpass", "freq_low": 2000, "freq_high": 5000, "desc": "Punchy mid-range"},
    "Smooth Jazz": {"type": "bandpass", "freq_low": 300, "freq_high": 5000, "desc": "Warm velvety tone"},
    "Electronic Punch": {"type": "bandstop", "freq_low": 800, "freq_high": 2000, "desc": "Enhance definition"},
    "Ambient Pad": {"type": "lowpass", "freq": 4000, "desc": "Soft ethereal quality"},
    "Metal Edge": {"type": "highpass", "freq": 3000, "desc": "Aggressive cutting highs"},
    "Intimate Vocal": {"type": "bandpass", "freq_low": 1000, "freq_high": 3500, "desc": "Close personal tone"},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_audio_from_mic():
    """Record audio from microphone"""
    try:
        audio_data = st.audio_input("🎤 Record your voice", key="mic_input")
        if audio_data is not None:
            audio_bytes = audio_data.getvalue()
            data, sr = sf.read(io.BytesIO(audio_bytes))
            return data, sr
    except:
        pass
    return None, None

def load_audio_from_file(uploaded_file):
    """Load audio from file"""
    try:
        if uploaded_file is not None:
            data, sr = sf.read(uploaded_file)
            return data, sr
    except Exception as e:
        st.error(f"❌ File error: {e}")
    return None, None

def preprocess_audio(audio_data, sr):
    """Normalize audio"""
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = np.atleast_1d(audio_data.flatten())
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    return audio_data, sr

def generate_test_signal(signal_type, duration=2.0, sr=16000):
    """Generate test signals"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if signal_type == 'Speech-like (Chirp + Noise)':
        chirp = signal.chirp(t, 200, duration, 2000, method='linear')
        noise = np.random.normal(0, 0.2, len(t))
        return 0.7 * chirp + 0.3 * noise
    elif signal_type == 'Pure Tone (440 Hz)':
        return 0.5 * np.sin(2 * np.pi * 440 * t)
    elif signal_type == 'Harmonic Complex (100 Hz)':
        harmonics = np.sin(2 * np.pi * 100 * t)
        for h in [2, 3, 4]:
            harmonics += 0.3 / h * np.sin(2 * np.pi * 100 * h * t)
        return harmonics / 3
    elif signal_type == 'White Noise':
        return np.random.normal(0, 0.3, len(t))
    else:  # Impulse
        sig = np.zeros_like(t)
        sig[int(0.1*sr)] = 1.0
        return sig
    return 0.5 * np.sin(2 * np.pi * 440 * t)

def generate_acoustic_ir(venue_name, sr, duration=1.0):
    """Generate impulse response for venues"""
    space = ACOUSTIC_SPACES.get(venue_name, ACOUSTIC_SPACES["Concert Hall"])
    num_samples = int(duration * sr)
    h = np.zeros(num_samples)
    h[0] = 1.0
    
    decay_rate = space["decay"] / 1000
    spacing_samples = int(space["spacing"] * sr)
    
    for i in range(space["reflections"]):
        delay = (i + 1) * spacing_samples
        if delay < len(h):
            amplitude = np.exp(-decay_rate * delay) * (1 + 0.05 * np.random.random())
            h[delay] = amplitude
    
    max_val = np.max(np.abs(h))
    if max_val > 0:
        h = h / max_val
    return h

def design_digital_filter(filter_type, order, cutoff, sr, filter_subtype='low'):
    """Design IIR filter"""
    nyquist = sr / 2
    
    if isinstance(cutoff, (tuple, list)):
        normalized_cutoff = [min(0.99, c / nyquist) for c in cutoff]
    else:
        normalized_cutoff = min(0.99, cutoff / nyquist)
    
    try:
        filter_dict = {"butterworth": butter, "cheby1": cheby1, "cheby2": cheby2, "ellip": ellip}
        if filter_type in ["cheby1", "cheby2", "ellip"]:
            if filter_type == "cheby1":
                sos = cheby1(order, 0.1, normalized_cutoff, btype=filter_subtype, output='sos')
            elif filter_type == "cheby2":
                sos = cheby2(order, 40, normalized_cutoff, btype=filter_subtype, output='sos')
            else:
                sos = ellip(order, 0.1, 40, normalized_cutoff, btype=filter_subtype, output='sos')
        else:
            sos = butter(order, normalized_cutoff, btype=filter_subtype, output='sos')
        return sos
    except:
        return None

def apply_filter(sig, sos):
    """Apply filter"""
    return sosfilt(sos, sig)

def compute_freq_response(sos, sr):
    """Compute frequency response"""
    w = np.logspace(0, np.log10(sr/2), 1024)
    w_rad = 2 * np.pi * w / sr
    w_rad, h = sosfreqz(sos, w_rad)
    return w_rad * sr / (2 * np.pi), np.abs(h), np.angle(h)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header-premium">
    <h1>🎵 LSI ACOUSTIC STUDIO PRO</h1>
    <div class="subtitle">Premium Interactive DSP Platform | World-Famous Venues | Professional Music Filters</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# ⚙️ AUDIO SETUP")
    
    if 'x' not in st.session_state:
        st.session_state.x = None
    if 'Fs' not in st.session_state:
        st.session_state.Fs = None
    
    audio_source = st.radio("Select Source", ["🎤 Microphone", "📁 Upload File", "🔧 Generate Test"], key="audio_src")
    
    if audio_source == "🎤 Microphone":
        x, Fs = record_audio_from_mic()
        if x is not None:
            st.session_state.x = x
            st.session_state.Fs = Fs
            st.success("✅ Recorded!")
    
    elif audio_source == "📁 Upload File":
        uploaded = st.file_uploader("Upload .wav", type=["wav"], key="file_upload")
        if uploaded:
            x, Fs = load_audio_from_file(uploaded)
            if x is not None:
                st.session_state.x = x
                st.session_state.Fs = Fs
                st.success("✅ Loaded!")
    
    else:
        signal_type = st.selectbox("Signal Type", ['Speech-like', 'Pure Tone (440 Hz)', 'Harmonic', 'White Noise', 'Impulse'], key="sig_type")
        if st.button("Generate", key="gen_sig", use_container_width=True):
            x = generate_test_signal(signal_type)
            Fs = 16000
            x, Fs = preprocess_audio(x, Fs)
            st.session_state.x = x
            st.session_state.Fs = Fs
            st.success("✅ Generated!")
    
    if st.session_state.x is not None:
        st.divider()
        st.markdown("### 📊 SIGNAL")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duration", f"{len(st.session_state.x)/st.session_state.Fs:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{st.session_state.Fs}Hz")
        st.audio(st.session_state.x, sample_rate=st.session_state.Fs)

# ============================================================================
# MAIN TABS
# ============================================================================

tabs = st.tabs(["🎛️ Convolution", "🎙️ Music Filters", "🏛️ World Venues", "🔧 Filters", "📊 Analysis", "✨ Effects"])

# ============================================================================
# TAB 1: CONVOLUTION
# ============================================================================

with tabs[0]:
    st.markdown('<h2 class="section-header">⚡ Convolution Engine</h2>', unsafe_allow_html=True)
    
    if st.session_state.x is None:
        st.warning("⚠️ Load audio from sidebar first!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 IR Design")
        ir_mode = st.radio("Type", ["Echo", "Room"], key="ir_mode_c", horizontal=True)
        
        if ir_mode == "Echo":
            delay = st.slider("Delay (sec)", 0.05, 1.0, 0.3)
            attenuation = st.slider("Attenuation", 0.1, 0.95, 0.5)
            num_echoes = st.slider("Echoes", 1, 5, 3)
            
            delay_samples = int(delay * st.session_state.Fs)
            h = np.zeros(delay_samples * (num_echoes + 1))
            h[0] = 1.0
            for i in range(1, num_echoes + 1):
                pos = i * delay_samples
                if pos < len(h):
                    h[pos] = attenuation ** i
        else:
            room = st.selectbox("Room", list(ACOUSTIC_SPACES.keys())[:5], key="room_c")
            h = generate_acoustic_ir(room, st.session_state.Fs, duration=1.0)
    
    with col2:
        st.markdown("### ⚙️ Options")
        apply_prefilter = st.checkbox("Pre-Filter", value=False)
        if apply_prefilter:
            ftype = st.selectbox("Filter", ["Butterworth", "Chebyshev I"], key="ftype_c")
            order = st.slider("Order", 2, 8, 4, key="order_c")
            cutoff = st.slider("Cutoff (Hz)", 100, 8000, 2000, key="cutoff_c")
            sos = design_digital_filter(ftype.lower().replace(" ", ""), order, cutoff, st.session_state.Fs)
            x_proc = apply_filter(st.session_state.x, sos) if sos else st.session_state.x.copy()
        else:
            x_proc = st.session_state.x.copy()
    
    st.divider()
    
    if st.button("🔊 COMPUTE CONVOLUTION", key="comp_conv", use_container_width=True, type="primary"):
        y = signal.convolve(x_proc, h, mode='full')
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / (max_val * 1.2)
        
        t_x = np.arange(len(x_proc)) / st.session_state.Fs
        t_h = np.arange(len(h)) / st.session_state.Fs
        t_y = np.arange(len(y)) / st.session_state.Fs
        
        # FIXED VISUALIZATION
        fig = plt.figure(figsize=(18, 14))
        fig.patch.set_facecolor('#0a0e27')
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # Step 1: Input
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t_x, x_proc, linewidth=3, color='#667eea', alpha=0.9)
        ax1.fill_between(t_x, x_proc, alpha=0.2, color='#667eea')
        ax1.axhline(y=0, color='white', linewidth=0.8, alpha=0.3, linestyle='--')
        ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
        ax1.set_title('STEP 1: Input Signal x[n]', fontsize=14, fontweight='bold', color='#667eea')
        ax1.grid(True, alpha=0.15, color='white')
        ax1.set_facecolor('#1a1f3a')
        ax1.tick_params(colors='white', labelsize=10)
        ax1.set_xlim([0, max(t_x) if len(t_x) > 0 else 1])
        
        # Step 2: IR
        ax2 = fig.add_subplot(gs[1, 0])
        markerline, stemlines, baseline = ax2.stem(t_h[:min(500, len(t_h))], h[:min(500, len(h))])
        stemlines.set_color('#43e97b')
        stemlines.set_linewidth(2.5)
        markerline.set_color('#43e97b')
        markerline.set_markersize(9)
        baseline.set_color('white')
        baseline.set_linewidth(0.8)
        baseline.set_alpha(0.3)
        ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
        ax2.set_title(f'STEP 2: Impulse Response h[n]', fontsize=13, fontweight='bold', color='#43e97b')
        ax2.grid(True, alpha=0.15, color='white')
        ax2.set_facecolor('#1a1f3a')
        ax2.tick_params(colors='white', labelsize=10)
        
        # Step 3: Output
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(t_y[:min(len(t_y), int(5*st.session_state.Fs))], y[:min(len(y), int(5*st.session_state.Fs))], linewidth=3, color='#f5576c', alpha=0.9)
        ax3.fill_between(t_y[:min(len(t_y), int(5*st.session_state.Fs))], y[:min(len(y), int(5*st.session_state.Fs))], alpha=0.2, color='#f5576c')
        ax3.axhline(y=0, color='white', linewidth=0.8, alpha=0.3, linestyle='--')
        ax3.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
        ax3.set_title('STEP 3: Output y[n] = x[n] * h[n]', fontsize=13, fontweight='bold', color='#f5576c')
        ax3.grid(True, alpha=0.15, color='white')
        ax3.set_facecolor('#1a1f3a')
        ax3.tick_params(colors='white', labelsize=10)
        
        # Step 4: Frequency Domain
        ax4 = fig.add_subplot(gs[2, :])
        freq_x = fftfreq(len(x_proc), 1/st.session_state.Fs)[:len(x_proc)//2]
        mag_x = np.abs(fft(x_proc))[:len(x_proc)//2]
        freq_y = fftfreq(len(y), 1/st.session_state.Fs)[:len(y)//2]
        mag_y = np.abs(fft(y))[:len(y)//2]
        
        ax4.semilogy(freq_x, mag_x + 1e-10, linewidth=2.5, color='#667eea', alpha=0.85, label='Input')
        ax4.semilogy(freq_y, mag_y + 1e-10, linewidth=2.5, color='#f5576c', alpha=0.85, label='Output')
        ax4.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold', color='white')
        ax4.set_ylabel('Magnitude', fontsize=12, fontweight='bold', color='white')
        ax4.set_title('STEP 4: Frequency Domain - Comb Filter Effect', fontsize=14, fontweight='bold', color='#667eea')
        ax4.grid(True, alpha=0.15, which='both', color='white')
        ax4.set_xlim([0, 8000])
        ax4.set_facecolor('#1a1f3a')
        ax4.tick_params(colors='white', labelsize=10)
        ax4.legend(fontsize=11, loc='upper right', facecolor='#1a1f3a', edgecolor='white')
        
        # Step 5: Signal Lengths
        ax5 = fig.add_subplot(gs[3, :])
        lengths = [len(x_proc), len(h), len(y)]
        bars = ax5.bar(['Input x[n]', 'IR h[n]', 'Output y[n]'], lengths, color=['#667eea', '#43e97b', '#f5576c'], alpha=0.8, edgecolor='white', linewidth=2)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontweight='bold', color='white', fontsize=11)
        ax5.set_ylabel('Length (samples)', fontsize=12, fontweight='bold', color='white')
        ax5.set_title('STEP 5: Signal Length Expansion', fontsize=14, fontweight='bold', color='#667eea')
        ax5.grid(True, alpha=0.15, axis='y', color='white')
        ax5.set_facecolor('#1a1f3a')
        ax5.tick_params(colors='white', labelsize=10)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        st.markdown('<h3 class="section-header">🔊 Audio Playback</h3>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 14px; text-align: center;"><p style="color: white; font-weight: 700; margin: 0;">🎵 Original</p></div>', unsafe_allow_html=True)
            st.audio(x_proc, sample_rate=st.session_state.Fs)
        with c2:
            st.markdown('<div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); padding: 1.5rem; border-radius: 14px; text-align: center;"><p style="color: white; font-weight: 700; margin: 0;">📢 Convolved</p></div>', unsafe_allow_html=True)
            st.audio(y, sample_rate=st.session_state.Fs)
        with c3:
            st.markdown('<div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1.5rem; border-radius: 14px; text-align: center;"><p style="color: white; font-weight: 700; margin: 0;">🎯 IR</p></div>', unsafe_allow_html=True)
            st.audio(h, sample_rate=st.session_state.Fs)
        
        st.divider()
        st.markdown('<h3 class="section-header">📊 Metrics</h3>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Input</div><div class="metric-value">{len(x_proc):,}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card accent2"><div class="metric-label">IR</div><div class="metric-value">{len(h):,}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card accent3"><div class="metric-label">Output</div><div class="metric-value">{len(y):,}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card accent4"><div class="metric-label">Peak</div><div class="metric-value">{np.max(np.abs(y)):.3f}</div></div>', unsafe_allow_html=True)

# ============================================================================
# TAB 2: MUSIC FILTERS
# ============================================================================

with tabs[1]:
    st.markdown('<h2 class="section-header">🎙️ Music Filters</h2>', unsafe_allow_html=True)
    
    if st.session_state.x is None:
        st.warning("⚠️ Load audio first!")
        st.stop()
    
    filter_name = st.selectbox("Select Filter", list(MUSIC_FILTERS.keys()), key="music_filt")
    st.markdown(f"**{MUSIC_FILTERS[filter_name]['desc']}**")
    
    st.divider()
    
    if st.button("🎵 APPLY FILTER", key="apply_music", use_container_width=True, type="primary"):
        cfg = MUSIC_FILTERS[filter_name]
        if cfg["type"] in ["lowpass", "highpass"]:
            sos = design_digital_filter("butterworth", 4, cfg["freq"], st.session_state.Fs, cfg["type"])
        else:
            sos = design_digital_filter("butterworth", 4, (cfg["freq_low"], cfg["freq_high"]), st.session_state.Fs, cfg["type"])
        
        filtered = apply_filter(st.session_state.x, sos) if sos else st.session_state.x
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor('#0a0e27')
        
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        
        axes[0, 0].plot(t[:min(8000, len(t))], st.session_state.x[:min(8000, len(st.session_state.x))], linewidth=2, color='#667eea', alpha=0.85)
        axes[0, 0].fill_between(t[:min(8000, len(t))], st.session_state.x[:min(8000, len(st.session_state.x))], alpha=0.15, color='#667eea')
        axes[0, 0].set_title('Original', fontsize=12, fontweight='bold', color='#667eea')
        axes[0, 0].set_facecolor('#1a1f3a')
        axes[0, 0].tick_params(colors='white')
        axes[0, 0].grid(True, alpha=0.15, color='white')
        axes[0, 0].set_ylabel('Amplitude', fontweight='bold', color='white')
        
        axes[0, 1].plot(t[:min(8000, len(t))], filtered[:min(8000, len(filtered))], linewidth=2, color='#f5576c', alpha=0.85)
        axes[0, 1].fill_between(t[:min(8000, len(t))], filtered[:min(8000, len(filtered))], alpha=0.15, color='#f5576c')
        axes[0, 1].set_title(f'{filter_name}', fontsize=12, fontweight='bold', color='#f5576c')
        axes[0, 1].set_facecolor('#1a1f3a')
        axes[0, 1].tick_params(colors='white')
        axes[0, 1].grid(True, alpha=0.15, color='white')
        axes[0, 1].set_ylabel('Amplitude', fontweight='bold', color='white')
        
        freq_orig = fftfreq(len(st.session_state.x), 1/st.session_state.Fs)[:len(st.session_state.x)//2]
        mag_orig = np.abs(fft(st.session_state.x))[:len(st.session_state.x)//2]
        freq_filt = fftfreq(len(filtered), 1/st.session_state.Fs)[:len(filtered)//2]
        mag_filt = np.abs(fft(filtered))[:len(filtered)//2]
        
        axes[1, 0].semilogy(freq_orig, mag_orig + 1e-10, linewidth=2.5, color='#667eea', alpha=0.85, label='Original')
        axes[1, 0].semilogy(freq_filt, mag_filt + 1e-10, linewidth=2.5, color='#f5576c', alpha=0.85, label='Filtered')
        axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', color='white')
        axes[1, 0].set_ylabel('Magnitude', fontsize=11, fontweight='bold', color='white')
        axes[1, 0].set_title('Spectrum Comparison', fontsize=12, fontweight='bold', color='#667eea')
        axes[1, 0].set_xlim([0, 8000])
        axes[1, 0].set_facecolor('#1a1f3a')
        axes[1, 0].tick_params(colors='white')
        axes[1, 0].grid(True, alpha=0.15, which='both', color='white')
        axes[1, 0].legend(fontsize=10, facecolor='#1a1f3a', edgecolor='white')
        
        if sos is not None:
            w, h_resp, phase = compute_freq_response(sos, st.session_state.Fs)
            axes[1, 1].plot(w, 20*np.log10(np.abs(h_resp) + 1e-10), linewidth=2.5, color='#43e97b', alpha=0.85)
            axes[1, 1].fill_between(w, 20*np.log10(np.abs(h_resp) + 1e-10), alpha=0.15, color='#43e97b')
            axes[1, 1].axhline(-3, color='#f5576c', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', color='white')
            axes[1, 1].set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold', color='white')
            axes[1, 1].set_title('Filter Response', fontsize=12, fontweight='bold', color='#667eea')
            axes[1, 1].set_xlim([0, 8000])
            axes[1, 1].set_facecolor('#1a1f3a')
            axes[1, 1].tick_params(colors='white')
            axes[1, 1].grid(True, alpha=0.15, which='both', color='white')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        st.markdown('<h3 class="section-header">🔊 Playback</h3>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 12px; text-align: center;"><p style="color: white; font-weight: 700; margin: 0;">Original</p></div>', unsafe_allow_html=True)
            st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
        with c2:
            st.markdown('<div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); padding: 1.2rem; border-radius: 12px; text-align: center;"><p style="color: white; font-weight: 700; margin: 0;">Filtered</p></div>', unsafe_allow_html=True)
            st.audio(filtered, sample_rate=st.session_state.Fs)

# ============================================================================
# TAB 3: WORLD VENUES
# ============================================================================

with tabs[2]:
    st.markdown('<h2 class="section-header">🏛️ World Venues</h2>', unsafe_allow_html=True)
    
    if st.session_state.x is None:
        st.warning("⚠️ Load audio first!")
        st.stop()
    
    venue = st.selectbox("Select Venue", list(ACOUSTIC_SPACES.keys()), key="venue_select")
    space = ACOUSTIC_SPACES[venue]
    st.markdown(f"**{space['emoji']} {space['desc']}**")
    
    st.divider()
    
    if st.button("🎵 SIMULATE", key="sim_venue", use_container_width=True, type="primary"):
        h = generate_acoustic_ir(venue, st.session_state.Fs, duration=1.0)
        y = signal.convolve(st.session_state.x, h, mode='full')
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / (max_val * 1.2)
        
        t_h = np.arange(len(h)) / st.session_state.Fs
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor('#0a0e27')
        
        # IR
        axes[0, 0].plot(t_h, h, linewidth=2, color='#43e97b', alpha=0.85)
        axes[0, 0].fill_between(t_h, h, alpha=0.15, color='#43e97b')
        axes[0, 0].set_title(f'IR - {venue}', fontsize=12, fontweight='bold', color='#667eea')
        axes[0, 0].set_facecolor('#1a1f3a')
        axes[0, 0].tick_params(colors='white')
        axes[0, 0].grid(True, alpha=0.15, color='white')
        axes[0, 0].set_ylabel('Amplitude', fontweight='bold', color='white')
        
        # Energy decay FIXED
        envelope = np.abs(h) + 1e-10
        energy_db = 20 * np.log10(envelope)
        axes[0, 1].plot(t_h, energy_db, linewidth=2.5, color='#f5576c', alpha=0.85)
        axes[0, 1].set_title('Energy Decay', fontsize=12, fontweight='bold', color='#667eea')
        axes[0, 1].set_facecolor('#1a1f3a')
        axes[0, 1].tick_params(colors='white')
        axes[0, 1].grid(True, alpha=0.15, color='white')
        axes[0, 1].set_ylabel('Energy (dB)', fontweight='bold', color='white')
        
        # Spectrum
        freq_h = fftfreq(len(h), 1/st.session_state.Fs)[:len(h)//2]
        mag_h = np.abs(fft(h))[:len(h)//2]
        axes[1, 0].semilogy(freq_h, mag_h + 1e-10, linewidth=2.5, color='#667eea', alpha=0.85)
        axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', color='white')
        axes[1, 0].set_ylabel('Magnitude', fontsize=11, fontweight='bold', color='white')
        axes[1, 0].set_title('Spectrum', fontsize=12, fontweight='bold', color='#667eea')
        axes[1, 0].set_xlim([0, 8000])
        axes[1, 0].set_facecolor('#1a1f3a')
        axes[1, 0].tick_params(colors='white')
        axes[1, 0].grid(True, alpha=0.15, which='both', color='white')
        
        # Cumulative energy FIXED - now correctly shows 0-100%
        energy_squared = h ** 2
        cumsum = np.cumsum(energy_squared) / np.sum(energy_squared) * 100
        axes[1, 1].plot(t_h, cumsum, linewidth=2.5, color='#f093fb', alpha=0.85)
        axes[1, 1].set_xlabel('Time (s)', fontsize=11, fontweight='bold', color='white')
        axes[1, 1].set_ylabel('Cumulative Energy (%)', fontsize=11, fontweight='bold', color='white')
        axes[1, 1].set_title('Energy Build-Up', fontsize=12, fontweight='bold', color='#667eea')
        axes[1, 1].set_ylim([0, 100])
        axes[1, 1].set_facecolor('#1a1f3a')
        axes[1, 1].tick_params(colors='white')
        axes[1, 1].grid(True, alpha=0.15, color='white')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
            st.caption("Original")
        with c2:
            st.audio(y, sample_rate=st.session_state.Fs)
            st.caption(f"In {venue}")

# ============================================================================
# TAB 4: ADVANCED FILTERS
# ============================================================================

with tabs[3]:
    st.markdown('<h2 class="section-header">🔧 Digital Filters</h2>', unsafe_allow_html=True)
    
    if st.session_state.x is None:
        st.warning("⚠️ Load audio first!")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ftype = st.selectbox("Type", ["butterworth", "cheby1", "cheby2", "ellip"], key="ftype_adv")
    with col2:
        subtype = st.selectbox("Mode", ["low", "high", "band", "stop"], key="subtype_adv")
    with col3:
        order = st.slider("Order", 2, 10, 5, key="order_adv")
    with col4:
        if ftype != "butterworth":
            ripple = st.slider("Ripple", 0.1, 3.0, 0.5, key="ripple_adv")
    
    st.divider()
    
    if subtype in ["low", "high"]:
        cutoff = st.slider("Cutoff (Hz)", 100, 7900, 2000, 50, key="cutoff_adv")
        sos = design_digital_filter(ftype, order, cutoff, st.session_state.Fs, subtype)
    else:
        f_low = st.slider("Low (Hz)", 100, 3900, 500, key="flow_adv")
        f_high = st.slider("High (Hz)", f_low+100, 8000, 4000, key="fhigh_adv")
        sos = design_digital_filter(ftype, order, (f_low, f_high), st.session_state.Fs, subtype)
    
    if sos:
        st.divider()
        w, mag, phase = compute_freq_response(sos, st.session_state.Fs)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 11))
        fig.patch.set_facecolor('#0a0e27')
        
        axes[0].plot(w, 20*np.log10(mag + 1e-10), linewidth=3, color='#667eea', alpha=0.85)
        axes[0].fill_between(w, 20*np.log10(mag + 1e-10), alpha=0.15, color='#667eea')
        axes[0].axhline(-3, color='#f5576c', linestyle='--', alpha=0.6)
        axes[0].set_ylabel('Magnitude (dB)', fontsize=12, fontweight='bold', color='white')
        axes[0].set_title('Magnitude Response', fontsize=13, fontweight='bold', color='#667eea')
        axes[0].set_xlim([0, 8000])
        axes[0].set_facecolor('#1a1f3a')
        axes[0].tick_params(colors='white')
        axes[0].grid(True, alpha=0.15, color='white')
        
        axes[1].plot(w, np.degrees(phase), linewidth=3, color='#43e97b', alpha=0.85)
        axes[1].set_ylabel('Phase (°)', fontsize=12, fontweight='bold', color='white')
        axes[1].set_title('Phase Response', fontsize=13, fontweight='bold', color='#667eea')
        axes[1].set_xlim([0, 8000])
        axes[1].set_facecolor('#1a1f3a')
        axes[1].tick_params(colors='white')
        axes[1].grid(True, alpha=0.15, color='white')
        
        gd = -np.diff(np.unwrap(phase)) / np.diff(2*np.pi*w/st.session_state.Fs)
        axes[2].plot(w[:-1], gd, linewidth=3, color='#f5576c', alpha=0.85)
        axes[2].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold', color='white')
        axes[2].set_ylabel('Group Delay (samples)', fontsize=12, fontweight='bold', color='white')
        axes[2].set_title('Group Delay', fontsize=13, fontweight='bold', color='#667eea')
        axes[2].set_xlim([0, 8000])
        axes[2].set_facecolor('#1a1f3a')
        axes[2].tick_params(colors='white')
        axes[2].grid(True, alpha=0.15, color='white')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        if st.button("APPLY FILTER", key="apply_adv", use_container_width=True, type="primary"):
            filtered = apply_filter(st.session_state.x, sos)
            c1, c2 = st.columns(2)
            with c1:
                st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
                st.caption("Original")
            with c2:
                st.audio(filtered, sample_rate=st.session_state.Fs)
                st.caption("Filtered")

# ============================================================================
# TAB 5: SIGNAL ANALYSIS
# ============================================================================

with tabs[4]:
    st.markdown('<h2 class="section-header">📊 Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.x is None:
        st.warning("⚠️ Load audio first!")
        st.stop()
    
    analysis = st.selectbox("Type", ["Spectrogram", "PSD", "Autocorr", "Zero Cross"], key="analysis")
    
    st.divider()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0a0e27')
    ax.set_facecolor('#1a1f3a')
    
    if analysis == "Spectrogram":
        f, t, Sxx = signal.spectrogram(st.session_state.x, st.session_state.Fs, nperseg=1024)
        im = ax.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='twilight_shifted')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold', color='white')
        ax.set_xlabel('Time (s)', fontweight='bold', color='white')
        ax.set_title('Spectrogram', fontsize=13, fontweight='bold', color='#667eea')
        ax.set_ylim([0, 8000])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontweight='bold')
    else:
        freqs, psd = signal.welch(st.session_state.x, st.session_state.Fs, nperseg=1024)
        ax.semilogy(freqs, psd, linewidth=2.5, color='#667eea', alpha=0.85)
        ax.fill_between(freqs, psd, alpha=0.15, color='#667eea')
        ax.set_xlabel('Frequency (Hz)', fontweight='bold', color='white')
        ax.set_ylabel('Power/Freq', fontweight='bold', color='white')
        ax.set_title('PSD', fontsize=13, fontweight='bold', color='#667eea')
        ax.set_xlim([0, 8000])
        ax.grid(True, alpha=0.15, which='both', color='white')
    
    ax.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# TAB 6: EFFECTS
# ============================================================================

with tabs[5]:
    st.markdown('<h2 class="section-header">✨ Effects</h2>', unsafe_allow_html=True)
    
    if st.session_state.x is None:
        st.warning("⚠️ Load audio first!")
        st.stop()
    
    effect = st.selectbox("Effect", ["Vibrato", "Tremolo", "Distortion", "Echo"], key="effect")
    
    st.divider()
    
    if effect == "Vibrato":
        rate = st.slider("Rate (Hz)", 2.0, 15.0, 5.0)
        depth = st.slider("Depth", 0.01, 0.5, 0.1)
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        mod = 1 + depth * np.sin(2*np.pi*rate*t)
        processed = st.session_state.x * mod
    elif effect == "Tremolo":
        rate = st.slider("Rate (Hz)", 2.0, 15.0, 5.0)
        depth = st.slider("Depth", 0.1, 1.0, 0.5)
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        mod = 1 - depth * (1 + np.sin(2*np.pi*rate*t)) / 2
        processed = st.session_state.x * mod
    elif effect == "Distortion":
        gain = st.slider("Gain", 1.0, 20.0, 5.0)
        processed = np.tanh(st.session_state.x * gain)
    else:  # Echo
        delay = st.slider("Delay (s)", 0.1, 1.0, 0.3)
        feedback = st.slider("Feedback", 0.0, 0.95, 0.5)
        delay_samples = int(delay * st.session_state.Fs)
        processed = np.zeros(len(st.session_state.x) + delay_samples * 3)
        processed[:len(st.session_state.x)] = st.session_state.x
        for i in range(3):
            pos = (i+1) * delay_samples
            processed[pos:pos+len(st.session_state.x)] += (feedback**(i+1)) * st.session_state.x
        processed = processed[:len(st.session_state.x) + delay_samples*3]
    
    if st.button("APPLY EFFECT", key="apply_eff", use_container_width=True, type="primary"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor('#0a0e27')
        
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        t_proc = np.arange(len(processed)) / st.session_state.Fs
        
        axes[0, 0].plot(t[:8000], st.session_state.x[:8000], linewidth=2, color='#667eea', alpha=0.85)
        axes[0, 0].fill_between(t[:8000], st.session_state.x[:8000], alpha=0.15, color='#667eea')
        axes[0, 0].set_title('Original', fontsize=12, fontweight='bold', color='#667eea')
        axes[0, 0].set_facecolor('#1a1f3a')
        axes[0, 0].tick_params(colors='white')
        axes[0, 0].grid(True, alpha=0.15, color='white')
        
        axes[0, 1].plot(t_proc[:min(8000, len(t_proc))], processed[:min(8000, len(processed))], linewidth=2, color='#f5576c', alpha=0.85)
        axes[0, 1].fill_between(t_proc[:min(8000, len(t_proc))], processed[:min(8000, len(processed))], alpha=0.15, color='#f5576c')
        axes[0, 1].set_title(f'{effect}', fontsize=12, fontweight='bold', color='#f5576c')
        axes[0, 1].set_facecolor('#1a1f3a')
        axes[0, 1].tick_params(colors='white')
        axes[0, 1].grid(True, alpha=0.15, color='white')
        
        freq_orig = fftfreq(len(st.session_state.x), 1/st.session_state.Fs)[:len(st.session_state.x)//2]
        mag_orig = np.abs(fft(st.session_state.x))[:len(st.session_state.x)//2]
        freq_proc = fftfreq(len(processed), 1/st.session_state.Fs)[:len(processed)//2]
        mag_proc = np.abs(fft(processed))[:len(processed)//2]
        
        axes[1, 0].semilogy(freq_orig, mag_orig + 1e-10, linewidth=2.5, color='#667eea', alpha=0.85, label='Original')
        axes[1, 0].semilogy(freq_proc, mag_proc + 1e-10, linewidth=2.5, color='#f5576c', alpha=0.85, label=effect)
        axes[1, 0].set_xlabel('Frequency (Hz)', fontweight='bold', color='white')
        axes[1, 0].set_ylabel('Magnitude', fontweight='bold', color='white')
        axes[1, 0].set_title('Spectrum', fontsize=12, fontweight='bold', color='#667eea')
        axes[1, 0].set_xlim([0, 8000])
        axes[1, 0].set_facecolor('#1a1f3a')
        axes[1, 0].tick_params(colors='white')
        axes[1, 0].grid(True, alpha=0.15, which='both', color='white')
        axes[1, 0].legend(fontsize=10, facecolor='#1a1f3a', edgecolor='white')
        
        f, t_spec, Sxx = signal.spectrogram(processed, st.session_state.Fs, nperseg=1024)
        im = axes[1, 1].pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='twilight_shifted')
        axes[1, 1].set_ylabel('Frequency (Hz)', fontweight='bold', color='white')
        axes[1, 1].set_title('Spectrogram', fontsize=12, fontweight='bold', color='#667eea')
        axes[1, 1].set_ylim([0, 8000])
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Power (dB)', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
            st.caption("Original")
        with c2:
            st.audio(processed, sample_rate=st.session_state.Fs)
            st.caption(f"With {effect}")

st.divider()
st.markdown("""
<div class="info-box">
    <h4 style="margin-top: 0;">💡 About LSI Acoustic Studio PRO</h4>
    <p style="margin-bottom: 0;">Premium interactive DSP platform with real-time signal processing, acoustic simulation, professional filters, and creative effects.</p>
</div>
""", unsafe_allow_html=True)
