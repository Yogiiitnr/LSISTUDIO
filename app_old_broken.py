# coding: utf-8
"""
LSI Acoustic Studio PRO - Advanced Interactive DSP Platform
Premium acoustic simulation with world-famous venues and professional UI/UX
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
# PREMIUM STYLING WITH MODERN UI/UX
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2a2540 100%);
        min-height: 100vh;
    }
    
    .main > div:first-child {
        padding: 2rem;
    }
    
    /* Premium Header */
    .header-premium {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.25), inset 0 1px 0 rgba(255,255,255,0.2);
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .header-premium::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
    }
    
    .header-premium h1 {
        margin: 0;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        line-height: 1;
    }
    
    .header-premium .subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 300;
        margin-top: 1rem;
        letter-spacing: 0.5px;
    }
    
    /* Floating Cards */
    .floating-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #16213e 100%);
        border: 1.5px solid rgba(102, 126, 234, 0.3);
        padding: 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(102,126,234,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .floating-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.35), inset 0 1px 0 rgba(102,126,234,0.1);
        border-color: rgba(102, 126, 234, 0.6);
    }
    
    /* Section Headers */
    .section-header {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid rgba(102, 126, 234, 0.4);
        position: relative;
    }
    
    .section-header::before {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 2px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.35);
    }
    
    .metric-label {
        font-size: 0.8rem;
        opacity: 0.9;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 900;
        margin-top: 1rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -1px;
    }
    
    .metric-card.accent1 { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .metric-card.accent2 { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .metric-card.accent3 { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .metric-card.accent4 { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid transparent !important;
        padding: 1rem 2.5rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.35) !important;
        text-transform: uppercase !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5) !important;
        border-color: rgba(255,255,255,0.3) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background: transparent;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        background: linear-gradient(135deg, #1a1f3a 0%, #16213e 100%) !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 2px solid #667eea !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Input Elements */
    .stSlider {
        background: transparent;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
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
        margin-top: 2rem;
        margin-bottom: 1.5rem;
    }
    
    p {
        color: #d0d0e0;
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(102, 126, 234, 0.08);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
    }
    
    /* Dividers */
    .divider {
        margin: 2rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ACOUSTIC SPACES DATABASE - FAMOUS LOCATIONS
# ============================================================================

ACOUSTIC_SPACES = {
    "Taj Mahal (India)": {
        "decay": 5.5,
        "reflections": 25,
        "spacing": 0.045,
        "description": "Marble mausoleum with warm, reverberant acoustics",
        "emoji": "🕌",
        "freq_response": {"bass": 1.2, "mid": 0.9, "treble": 1.1}
    },
    "Sydney Opera House": {
        "decay": 3.5,
        "reflections": 18,
        "spacing": 0.035,
        "description": "Modern concert venue with precise acoustic design",
        "emoji": "🎭",
        "freq_response": {"bass": 1.0, "mid": 1.1, "treble": 1.2}
    },
    "Pantheon (Rome)": {
        "decay": 8.0,
        "reflections": 35,
        "spacing": 0.055,
        "description": "Dome with exceptional long reverberation",
        "emoji": "🏛️",
        "freq_response": {"bass": 1.3, "mid": 0.85, "treble": 0.95}
    },
    "Grand Central Terminal": {
        "decay": 4.2,
        "reflections": 22,
        "spacing": 0.042,
        "description": "Iconic station with natural echo chambers",
        "emoji": "🚂",
        "freq_response": {"bass": 1.15, "mid": 0.95, "treble": 1.0}
    },
    "Ancient Pagoda (Japan)": {
        "decay": 2.8,
        "reflections": 15,
        "spacing": 0.038,
        "description": "Wooden structure with warm, intimate acoustics",
        "emoji": "🏯",
        "freq_response": {"bass": 1.1, "mid": 1.15, "treble": 0.85}
    },
    "Church of Santo Domingo": {
        "decay": 6.5,
        "reflections": 30,
        "spacing": 0.048,
        "description": "Cathedral with rich, singing space",
        "emoji": "⛪",
        "freq_response": {"bass": 1.2, "mid": 0.9, "treble": 1.15}
    },
    "Sagano Bamboo Grove (Japan)": {
        "decay": 1.5,
        "reflections": 8,
        "spacing": 0.025,
        "description": "Natural bamboo acoustic filter - tight, dry",
        "emoji": "🎋",
        "freq_response": {"bass": 0.85, "mid": 0.9, "treble": 1.3}
    },
    "Colosseum (Rome)": {
        "decay": 7.2,
        "reflections": 40,
        "spacing": 0.052,
        "description": "Ancient amphitheater with expansive reverb",
        "emoji": "🏛️",
        "freq_response": {"bass": 1.25, "mid": 0.88, "treble": 1.05}
    },
    "Concert Hall Acoustics": {
        "decay": 2.2,
        "reflections": 12,
        "spacing": 0.032,
        "description": "Optimized for musical performances",
        "emoji": "🎼",
        "freq_response": {"bass": 1.0, "mid": 1.0, "treble": 1.0}
    },
    "Underwater Cave (Hawaii)": {
        "decay": 3.8,
        "reflections": 20,
        "spacing": 0.044,
        "description": "Water-dampened unique reflections",
        "emoji": "🌊",
        "freq_response": {"bass": 0.95, "mid": 1.0, "treble": 0.9}
    },
    "Grand Canyon Echo": {
        "decay": 9.0,
        "reflections": 45,
        "spacing": 0.06,
        "description": "Massive canyon with natural long reverb",
        "emoji": "🏜️",
        "freq_response": {"bass": 1.3, "mid": 0.85, "treble": 1.0}
    },
    "Abbey Road Studio": {
        "decay": 1.8,
        "reflections": 10,
        "spacing": 0.028,
        "description": "Professional recording studio - controlled, warm",
        "emoji": "🎙️",
        "freq_response": {"bass": 0.95, "mid": 1.1, "treble": 0.95}
    }
}

# ============================================================================
# MUSIC FILTERS
# ============================================================================

MUSIC_FILTERS = {
    "Vocal Enhancer": {
        "type": "bandpass",
        "freq_low": 500,
        "freq_high": 4000,
        "description": "Boost vocal presence and clarity"
    },
    "Bass Booster": {
        "type": "lowpass",
        "freq": 200,
        "gain": 1.5,
        "description": "Enhance deep bass frequencies"
    },
    "Treble Shimmer": {
        "type": "highpass",
        "freq": 5000,
        "gain": 1.3,
        "description": "Bright, airy, shimmering highs"
    },
    "Lo-Fi Hip Hop": {
        "type": "bandpass",
        "freq_low": 100,
        "freq_high": 6000,
        "description": "Warm, compressed, retro sound"
    },
    "Presence Peak": {
        "type": "bandpass",
        "freq_low": 2000,
        "freq_high": 5000,
        "description": "Punchy mid-range presence"
    },
    "Smooth Jazz": {
        "type": "bandpass",
        "freq_low": 300,
        "freq_high": 5000,
        "description": "Warm, velvety acoustic tone"
    },
    "Electronic Punch": {
        "type": "bandstop",
        "freq_low": 800,
        "freq_high": 2000,
        "description": "Remove muddy frequencies, enhance definition"
    },
    "Ambient Pad": {
        "type": "lowpass",
        "freq": 4000,
        "description": "Soft, ethereal, pad-like quality"
    },
    "Metal Edge": {
        "type": "highpass",
        "freq": 3000,
        "gain": 1.5,
        "description": "Aggressive, cutting, sharp highs"
    },
    "Intimate Vocal": {
        "type": "bandpass",
        "freq_low": 1000,
        "freq_high": 3500,
        "description": "Close, personal vocal tone"
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_audio_from_mic():
    """Record audio from microphone"""
    audio_data = st.audio_input("🎤 Record your voice")
    if audio_data is not None:
        try:
            audio_bytes = audio_data.getvalue()
            data, sr = sf.read(io.BytesIO(audio_bytes))
            return data, sr
        except:
            return None, None
    return None, None

def load_audio_from_file(uploaded_file):
    """Load audio from file"""
    if uploaded_file is not None:
        try:
            data, sr = sf.read(uploaded_file)
            return data, sr
        except Exception as e:
            st.error(f"❌ File error: {e}")
            return None, None
    return None, None

def preprocess_audio(audio_data, sr):
    """Normalize and convert audio"""
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = np.atleast_1d(audio_data.flatten())
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    return audio_data, sr

def generate_test_signal(signal_type, duration=2.0, sr=16000):
    """Generate various test signals"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if signal_type == 'Speech-like (Chirp + Noise)':
        chirp = signal.chirp(t, 200, duration, 2000, method='linear')
        noise = np.random.normal(0, 0.2, len(t))
        return 0.7 * chirp + 0.3 * noise
    elif signal_type == 'Pure Tone (440 Hz - A4)':
        return 0.5 * np.sin(2 * np.pi * 440 * t)
    elif signal_type == 'Harmonic Complex (100 Hz)':
        harmonics = np.sin(2 * np.pi * 100 * t)
        for h in [2, 3, 4]:
            harmonics += 0.3 / h * np.sin(2 * np.pi * 100 * h * t)
        return harmonics / 3
    elif signal_type == 'White Noise':
        return np.random.normal(0, 0.3, len(t))
    elif signal_type == 'Impulse':
        sig = np.zeros_like(t)
        sig[int(0.1*sr)] = 1.0
        return sig
    return 0.5 * np.sin(2 * np.pi * 440 * t)

def generate_acoustic_ir(venue_name, sr, duration=1.0):
    """Generate impulse response for famous acoustic spaces"""
    space = ACOUSTIC_SPACES.get(venue_name, ACOUSTIC_SPACES["Concert Hall Acoustics"])
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
    """Design digital IIR filter"""
    nyquist = sr / 2
    
    if isinstance(cutoff, (tuple, list)):
        normalized_cutoff = [c / nyquist for c in cutoff]
    else:
        normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff >= 1 if not isinstance(normalized_cutoff, list) else any(f >= 1 for f in normalized_cutoff):
        normalized_cutoff = 0.99 if not isinstance(normalized_cutoff, list) else [min(0.99, f) for f in normalized_cutoff]
    
    try:
        if filter_type == 'butterworth':
            sos = butter(order, normalized_cutoff, btype=filter_subtype, output='sos')
        elif filter_type == 'cheby1':
            sos = cheby1(order, 0.1, normalized_cutoff, btype=filter_subtype, output='sos')
        elif filter_type == 'cheby2':
            sos = cheby2(order, 40, normalized_cutoff, btype=filter_subtype, output='sos')
        elif filter_type == 'ellip':
            sos = ellip(order, 0.1, 40, normalized_cutoff, btype=filter_subtype, output='sos')
        return sos
    except:
        return None

def apply_filter(sig, sos):
    """Apply filter to signal"""
    return sosfilt(sos, sig)

def compute_freq_response(sos, sr):
    """Compute filter frequency response"""
    w = np.logspace(0, np.log10(sr/2), 1024)
    w_rad = 2 * np.pi * w / sr
    w_rad, h = sosfreqz(sos, w_rad)
    return w_rad * sr / (2 * np.pi), np.abs(h), np.angle(h)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header-premium">
    <h1>🎵 LSI Acoustic Studio PRO</h1>
    <div class="subtitle">Premium Interactive DSP Platform with World-Famous Venues</div>
    <div class="subtitle" style="font-size: 0.95rem; opacity: 0.85; margin-top: 0.8rem;">Advanced signal processing, acoustic simulation, and music production tools</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - GLOBAL AUDIO INPUT
# ============================================================================

with st.sidebar:
    st.markdown("# ⚙️ AUDIO SETUP")
    
    # Initialize session state
    if 'x' not in st.session_state:
        st.session_state.x = None
    if 'Fs' not in st.session_state:
        st.session_state.Fs = None
    
    audio_source = st.radio(
        "Audio Source",
        ["🎤 Microphone", "📁 Upload File", "🔧 Generate Test"],
        key="audio_source"
    )
    
    if audio_source == "🎤 Microphone":
        x, Fs = record_audio_from_mic()
        if x is not None:
            st.session_state.x = x
            st.session_state.Fs = Fs
            st.success("✅ Microphone recorded!")
    
    elif audio_source == "📁 Upload File":
        uploaded = st.file_uploader("Upload .wav file", type=["wav"])
        if uploaded:
            x, Fs = load_audio_from_file(uploaded)
            if x is not None:
                st.session_state.x = x
                st.session_state.Fs = Fs
                st.success("✅ File loaded!")
    
    else:
        signal_type = st.selectbox("Test Signal",
            ['Speech-like (Chirp + Noise)', 'Pure Tone (440 Hz - A4)', 
             'Harmonic Complex (100 Hz)', 'White Noise', 'Impulse'])
        
        if st.button("Generate Signal", width='stretch'):
            x = generate_test_signal(signal_type)
            Fs = 16000
            x, Fs = preprocess_audio(x, Fs)
            st.session_state.x = x
            st.session_state.Fs = Fs
            st.success("✅ Test signal generated!")
    
    # Show loaded signal info
    if st.session_state.x is not None and st.session_state.Fs is not None:
        st.divider()
        st.markdown("### 📊 SIGNAL INFO")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duration", f"{len(st.session_state.x)/st.session_state.Fs:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{st.session_state.Fs}Hz")
        
        st.markdown("**Preview:**")
        st.audio(st.session_state.x, sample_rate=st.session_state.Fs)

# ============================================================================
# MAIN TABS
# ============================================================================

tabs = st.tabs([
    "🎛️ Convolution Engine",
    "🎙️ Music Filters",
    "🏛️ World Venues",
    "🔧 Advanced Filters",
    "📊 Signal Analysis",
    "✨ Effects Lab"
])

# ============================================================================
# TAB 1: CONVOLUTION ENGINE
# ============================================================================

with tabs[0]:
    st.markdown('<div class="section-header">⚡ Convolution Engine</div>', unsafe_allow_html=True)
    st.markdown("*Real-time impulse response convolution with signal visualization*")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header" style="font-size: 1.4rem; margin-top: 0;">📍 IR Design</div>', unsafe_allow_html=True)
        
        ir_mode = st.radio("Type", ["Echo", "Room Preset"], key="ir_mode_conv", horizontal=True)
        
        if ir_mode == "Echo":
            delay = st.slider("Delay (sec)", 0.05, 1.0, 0.3, 0.05)
            attenuation = st.slider("Attenuation", 0.1, 0.95, 0.5, 0.05)
            num_echoes = st.slider("Number of Echoes", 1, 5, 3)
            
            delay_samples = int(delay * st.session_state.Fs)
            h = np.zeros(delay_samples * (num_echoes + 1))
            h[0] = 1.0
            for i in range(1, num_echoes + 1):
                pos = i * delay_samples
                if pos < len(h):
                    h[pos] = attenuation ** i
        else:
            room = st.selectbox("Room Preset", 
                ["Small Room", "Concert Hall", "Cathedral"],
                key="room_conv")
            h = generate_acoustic_ir(room, st.session_state.Fs, duration=1.0)
    
    with col2:
        st.markdown('<div class="section-header" style="font-size: 1.4rem; margin-top: 0;">⚙️ Options</div>', unsafe_allow_html=True)
        
        apply_prefilter = st.checkbox("Apply Pre-Filter", value=False)
        
        if apply_prefilter:
            filter_type = st.selectbox("Filter Type", ["Butterworth", "Chebyshev I"])
            order = st.slider("Order", 2, 8, 4)
            cutoff = st.slider("Cutoff (Hz)", 100, 8000, 2000)
            
            filter_dict = {"Butterworth": "butterworth", "Chebyshev I": "cheby1"}
            sos = design_digital_filter(filter_dict[filter_type], order, cutoff, st.session_state.Fs)
            x_proc = apply_filter(st.session_state.x, sos) if sos else st.session_state.x.copy()
        else:
            x_proc = st.session_state.x.copy()
    
    st.divider()
    
    if st.button("🔊 COMPUTE CONVOLUTION", key="compute_conv", type="primary", use_container_width=True):
        y = signal.convolve(x_proc, h, mode='full')
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / (max_val * 1.2)
        
        t_x = np.arange(len(x_proc)) / st.session_state.Fs
        t_h = np.arange(len(h)) / st.session_state.Fs
        t_y = np.arange(len(y)) / st.session_state.Fs
        
        # Main visualization with signal processing flow
        fig = plt.figure(figsize=(18, 14))
        fig.patch.set_facecolor('#0a0e27')
        gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # Input signal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t_x, x_proc, linewidth=3, color='#667eea', alpha=0.95, label='Input Signal x[n]')
        ax1.fill_between(t_x, x_proc, alpha=0.25, color='#667eea')
        ax1.axhline(y=0, color='white', linewidth=0.8, alpha=0.3, linestyle='--')
        ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
        ax1.set_title('STEP 1: Input Signal x[n] - Time Domain', fontsize=14, fontweight='bold', color='#667eea')
        ax1.grid(True, alpha=0.15, linestyle='--', color='white')
        ax1.set_facecolor('#1a1f3a')
        ax1.tick_params(colors='white', labelsize=10)
        ax1.legend(loc='upper right', fontsize=11, facecolor='#1a1f3a', edgecolor='white', framealpha=0.9)
        ax1.set_xlim([0, max(t_x)])
        
        # Impulse response
        ax2 = fig.add_subplot(gs[1, 0])
        markerline, stemlines, baseline = ax2.stem(t_h[:min(500, len(t_h))], h[:min(500, len(h))], basefmt=' ')
        stemlines.set_color('#43e97b')
        stemlines.set_linewidth(2.5)
        markerline.set_color('#43e97b')
        markerline.set_markersize(9)
        baseline.set_color('white')
        baseline.set_linewidth(0.8)
        baseline.set_alpha(0.3)
        ax2.axhline(y=0, color='white', linewidth=0.8, alpha=0.3, linestyle='--')
        ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
        ax2.set_title(f'STEP 2: Impulse Response h[n] - {ir_mode}', fontsize=13, fontweight='bold', color='#43e97b')
        ax2.grid(True, alpha=0.15, color='white')
        ax2.set_facecolor('#1a1f3a')
        ax2.tick_params(colors='white', labelsize=10)
        ax2.set_xlim([0, min(t_h[-1] if len(t_h) > 0 else 0.5, 0.5)])
        
        # Output signal
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(t_y, y, linewidth=3, color='#f5576c', alpha=0.95, label='Output y[n] = x[n] * h[n]')
        ax3.fill_between(t_y, y, alpha=0.25, color='#f5576c')
        ax3.axhline(y=0, color='white', linewidth=0.8, alpha=0.3, linestyle='--')
        ax3.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
        ax3.set_title('STEP 3: Output Signal y[n] - Time Domain', fontsize=13, fontweight='bold', color='#f5576c')
        ax3.grid(True, alpha=0.15, color='white')
        ax3.set_xlim([0, min(t_y[-1] if len(t_y) > 0 else 5, 5)])
        ax3.set_facecolor('#1a1f3a')
        ax3.tick_params(colors='white', labelsize=10)
        ax3.legend(loc='upper right', fontsize=10, facecolor='#1a1f3a', edgecolor='white', framealpha=0.9)
        
        # Frequency domain analysis
        ax4 = fig.add_subplot(gs[2, :])
        freq_x, mag_x = fftfreq(len(x_proc), 1/st.session_state.Fs)[:len(x_proc)//2], np.abs(fft(x_proc))[:len(x_proc)//2]
        freq_y, mag_y = fftfreq(len(y), 1/st.session_state.Fs)[:len(y)//2], np.abs(fft(y))[:len(y)//2]
        
        mag_x_db = 20*np.log10(mag_x + 1e-10)
        mag_y_db = 20*np.log10(mag_y + 1e-10)
        
        ax4.semilogy(freq_x, mag_x_db, label='Input Spectrum', linewidth=2.5, color='#667eea', alpha=0.85)
        ax4.semilogy(freq_y, mag_y_db, label='Output Spectrum (Convolved)', linewidth=2.5, color='#f5576c', alpha=0.85)
        ax4.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold', color='white')
        ax4.set_ylabel('Magnitude (dB)', fontsize=12, fontweight='bold', color='white')
        ax4.set_title('STEP 4: Frequency Domain - Comb Filter Effect (Peaks & Nulls)', fontsize=14, fontweight='bold', color='#667eea')
        ax4.grid(True, alpha=0.15, which='both', color='white', linestyle='--')
        ax4.set_xlim([0, 8000])
        ax4.set_facecolor('#1a1f3a')
        ax4.tick_params(colors='white', labelsize=10)
        ax4.legend(fontsize=11, loc='upper right', facecolor='#1a1f3a', edgecolor='white', framealpha=0.95)
        ax4.axhline(y=0, color='white', linewidth=0.8, alpha=0.2, linestyle=':')
        
        # Time-frequency waterfall
        ax5 = fig.add_subplot(gs[3, :])
        ax5.bar(np.arange(3), [len(x_proc), len(h), len(y)], color=['#667eea', '#43e97b', '#f5576c'], 
                alpha=0.7, edgecolor='white', linewidth=2)
        ax5.set_xticks(np.arange(3))
        ax5.set_xticklabels(['Input x[n]', 'IR h[n]', 'Output y[n]'], fontsize=11, fontweight='bold', color='white')
        ax5.set_ylabel('Length (samples)', fontsize=12, fontweight='bold', color='white')
        ax5.set_title('STEP 5: Signal Length Comparison - Convolution Expansion', fontsize=14, fontweight='bold', color='#667eea')
        ax5.grid(True, alpha=0.15, axis='y', color='white')
        ax5.set_facecolor('#1a1f3a')
        ax5.tick_params(colors='white', labelsize=10)
        
        # Add value labels on bars
        for i, v in enumerate([len(x_proc), len(h), len(y)]):
            ax5.text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold', color='white', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        st.markdown('<div class="section-header">🔊 Audio Playback</div>', unsafe_allow_html=True)
        
        audio_col1, audio_col2, audio_col3 = st.columns(3)
        with audio_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.8rem; border-radius: 15px; text-align: center; margin-bottom: 1rem;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.15rem; margin: 0;">🎵 Original Input</p>
            </div>
            """, unsafe_allow_html=True)
            st.audio(x_proc, sample_rate=st.session_state.Fs)
            st.caption(f"Duration: {len(x_proc)/st.session_state.Fs:.2f}s | {st.session_state.Fs}Hz")
        with audio_col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); 
                        padding: 1.8rem; border-radius: 15px; text-align: center; margin-bottom: 1rem;
                        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.15rem; margin: 0;">📢 Convolved Output</p>
            </div>
            """, unsafe_allow_html=True)
            st.audio(y, sample_rate=st.session_state.Fs)
            st.caption(f"Duration: {len(y)/st.session_state.Fs:.2f}s | Normalized")
        with audio_col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 1.8rem; border-radius: 15px; text-align: center; margin-bottom: 1rem;
                        box-shadow: 0 10px 30px rgba(67, 233, 123, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.15rem; margin: 0;">🎯 Impulse Response</p>
            </div>
            """, unsafe_allow_html=True)
            st.audio(h, sample_rate=st.session_state.Fs)
            st.caption(f"Duration: {len(h)/st.session_state.Fs:.2f}s")
        
        st.divider()
        st.markdown('<div class="section-header">📊 Metrics & Analysis</div>', unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Input Length</div><div class="metric-value">{len(x_proc):,}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card accent2"><div class="metric-label">IR Length</div><div class="metric-value">{len(h):,}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card accent3"><div class="metric-label">Output Length</div><div class="metric-value">{len(y):,}</div></div>', unsafe_allow_html=True)
        with m4:
            peak_val = np.max(np.abs(y))
            st.markdown(f'<div class="metric-card accent4"><div class="metric-label">Peak Amplitude</div><div class="metric-value">{peak_val:.3f}</div></div>', unsafe_allow_html=True)
        
        st.divider()
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**⏱️ Time Domain Properties**")
            td_data = {
                "Property": ["Input Duration", "Output Duration", "Expansion", "Input RMS", "Output RMS"],
                "Value": [
                    f"{len(x_proc)/st.session_state.Fs:.3f}s",
                    f"{len(y)/st.session_state.Fs:.3f}s",
                    f"{len(y)/len(x_proc):.2f}x",
                    f"{np.sqrt(np.mean(x_proc**2)):.4f}",
                    f"{np.sqrt(np.mean(y**2)):.4f}"
                ]
            }
            st.dataframe(td_data, use_container_width=True, hide_index=True)
        
        with col_a2:
            st.markdown("**🎵 Frequency Domain Properties**")
            fd_data = {
                "Property": ["Input Peak Freq", "Output Peak Freq", "Input Max Mag", "Output Max Mag", "Magnification"],
                "Value": [
                    f"{freq_x[np.argmax(mag_x)]:.1f} Hz",
                    f"{freq_y[np.argmax(mag_y)]:.1f} Hz",
                    f"{np.max(mag_x):.2e}",
                    f"{np.max(mag_y):.2e}",
                    f"{np.max(mag_y)/np.max(mag_x):.2f}x"
                ]
            }
            st.dataframe(fd_data, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 2: MUSIC FILTERS
# ============================================================================

with tabs[1]:
    st.markdown('<div class="section-header">🎙️ Music Filters</div>', unsafe_allow_html=True)
    st.markdown("*Professional-grade pre-defined filters for music production*")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    selected_filter = st.selectbox(
        "Select Filter Preset",
        list(MUSIC_FILTERS.keys()),
        key="music_filter"
    )
    
    filter_info = MUSIC_FILTERS[selected_filter]
    st.markdown(f"**Description:** {filter_info['description']}")
    
    st.divider()
    
    if st.button("🎵 APPLY MUSIC FILTER", key="apply_music_filter", type="primary", use_container_width=True):
        filter_cfg = MUSIC_FILTERS[selected_filter]
        
        try:
            if filter_cfg["type"] in ["lowpass", "highpass"]:
                sos = design_digital_filter("butterworth", 4, filter_cfg["freq"], st.session_state.Fs, filter_cfg["type"])
            else:  # bandpass, bandstop
                sos = design_digital_filter("butterworth", 4, 
                                          (filter_cfg["freq_low"], filter_cfg["freq_high"]), 
                                          st.session_state.Fs, filter_cfg["type"])
            
            filtered = apply_filter(st.session_state.x, sos) if sos else st.session_state.x
            
            # Visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.patch.set_facecolor('#0a0e27')
            
            t = np.arange(len(st.session_state.x)) / st.session_state.Fs
            t_filt = np.arange(len(filtered)) / st.session_state.Fs
            
            # Time domain - original
            axes[0, 0].plot(t[:min(8000, len(t))], st.session_state.x[:min(8000, len(st.session_state.x))], 
                           linewidth=2, color='#667eea', alpha=0.85)
            axes[0, 0].fill_between(t[:min(8000, len(t))], st.session_state.x[:min(8000, len(st.session_state.x))], 
                                   alpha=0.15, color='#667eea')
            axes[0, 0].set_title('Original Signal (Time Domain)', fontsize=12, fontweight='bold', color='#667eea')
            axes[0, 0].set_ylabel('Amplitude', fontsize=11, fontweight='bold', color='white')
            axes[0, 0].grid(True, alpha=0.15, color='white')
            axes[0, 0].set_facecolor('#1a1f3a')
            axes[0, 0].tick_params(colors='white')
            
            # Time domain - filtered
            axes[0, 1].plot(t_filt[:min(8000, len(t_filt))], filtered[:min(8000, len(filtered))], 
                           linewidth=2, color='#f5576c', alpha=0.85)
            axes[0, 1].fill_between(t_filt[:min(8000, len(t_filt))], filtered[:min(8000, len(filtered))], 
                                   alpha=0.15, color='#f5576c')
            axes[0, 1].set_title(f'{selected_filter} - Filtered (Time Domain)', fontsize=12, fontweight='bold', color='#f5576c')
            axes[0, 1].set_ylabel('Amplitude', fontsize=11, fontweight='bold', color='white')
            axes[0, 1].grid(True, alpha=0.15, color='white')
            axes[0, 1].set_facecolor('#1a1f3a')
            axes[0, 1].tick_params(colors='white')
            
            # Frequency response
            freq_orig = fftfreq(len(st.session_state.x), 1/st.session_state.Fs)[:len(st.session_state.x)//2]
            mag_orig = np.abs(fft(st.session_state.x))[:len(st.session_state.x)//2]
            freq_filt = fftfreq(len(filtered), 1/st.session_state.Fs)[:len(filtered)//2]
            mag_filt = np.abs(fft(filtered))[:len(filtered)//2]
            
            mag_orig_db = 20*np.log10(mag_orig + 1e-10)
            mag_filt_db = 20*np.log10(mag_filt + 1e-10)
            
            axes[1, 0].semilogy(freq_orig, mag_orig_db, label='Original', linewidth=2.5, color='#667eea', alpha=0.85)
            axes[1, 0].semilogy(freq_filt, mag_filt_db, label=selected_filter, linewidth=2.5, color='#f5576c', alpha=0.85)
            axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', color='white')
            axes[1, 0].set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold', color='white')
            axes[1, 0].set_title('Frequency Response Comparison', fontsize=12, fontweight='bold', color='#667eea')
            axes[1, 0].grid(True, alpha=0.15, which='both', color='white')
            axes[1, 0].set_xlim([0, 8000])
            axes[1, 0].set_facecolor('#1a1f3a')
            axes[1, 0].tick_params(colors='white')
            axes[1, 0].legend(fontsize=10, facecolor='#1a1f3a', edgecolor='white', framealpha=0.9)
            
            # Filter response
            if sos is not None:
                w, h_resp, phase = compute_freq_response(sos, st.session_state.Fs)
                axes[1, 1].plot(w, 20*np.log10(np.abs(h_resp) + 1e-10), linewidth=2.5, color='#43e97b')
                axes[1, 1].fill_between(w, 20*np.log10(np.abs(h_resp) + 1e-10), alpha=0.2, color='#43e97b')
                axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', color='white')
                axes[1, 1].set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold', color='white')
                axes[1, 1].set_title('Filter Frequency Response', fontsize=12, fontweight='bold', color='#667eea')
                axes[1, 1].grid(True, alpha=0.15, which='both', color='white')
                axes[1, 1].axhline(-3, color='#f5576c', linestyle='--', alpha=0.5, label='-3dB')
                axes[1, 1].set_xlim([0, 8000])
                axes[1, 1].set_facecolor('#1a1f3a')
                axes[1, 1].tick_params(colors='white')
                axes[1, 1].legend(fontsize=10, facecolor='#1a1f3a', edgecolor='white', framealpha=0.9)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.divider()
            st.markdown('<div class="section-header">🔊 Audio Playback</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
                    <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">Original</p>
                </div>
                """, unsafe_allow_html=True)
                st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
            with c2:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
                    <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">Filtered</p>
                </div>
                """, unsafe_allow_html=True)
                st.audio(filtered, sample_rate=st.session_state.Fs)
        
        except Exception as e:
            st.error(f"❌ Error applying filter: {e}")

# ============================================================================
# TAB 3: WORLD VENUES
# ============================================================================

with tabs[2]:
    st.markdown('<div class="section-header">🏛️ World Famous Venues</div>', unsafe_allow_html=True)
    st.markdown("*Simulate acoustics of iconic locations around the world*")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    # Display venue grid
    st.markdown("### Select a Venue:")
    
    venue_cols = st.columns(3)
    venue_list = list(ACOUSTIC_SPACES.keys())
    
    selected_venue = None
    for i, venue_name in enumerate(venue_list):
        col_idx = i % 3
        with venue_cols[col_idx]:
            space = ACOUSTIC_SPACES[venue_name]
            if st.button(f"{space['emoji']} {venue_name}", key=f"venue_{i}", use_container_width=True):
                selected_venue = venue_name
    
    # Alternative: Selectbox
    selected_venue = st.selectbox("Or select from dropdown:", venue_list, key="venue_select")
    
    if selected_venue:
        space = ACOUSTIC_SPACES[selected_venue]
        
        st.divider()
        st.markdown(f"### {space['emoji']} {selected_venue}")
        st.markdown(f"*{space['description']}*")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("Decay Time", f"{space['decay']}s")
        with col_info2:
            st.metric("Reflections", f"{space['reflections']}")
        
        st.divider()
        
        if st.button("🎵 SIMULATE VENUE ACOUSTICS", key="simulate_venue", type="primary", use_container_width=True):
            h = generate_acoustic_ir(selected_venue, st.session_state.Fs, duration=1.0)
            y = signal.convolve(st.session_state.x, h, mode='full')
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / (max_val * 1.2)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.patch.set_facecolor('#0a0e27')
            
            t_h = np.arange(len(h)) / st.session_state.Fs
            t_y = np.arange(len(y)) / st.session_state.Fs
            
            # IR
            axes[0, 0].plot(t_h, h, linewidth=2, color='#43e97b', alpha=0.85)
            axes[0, 0].fill_between(t_h, h, alpha=0.15, color='#43e97b')
            axes[0, 0].set_ylabel('Amplitude', fontweight='bold', color='white')
            axes[0, 0].set_title(f'Impulse Response - {selected_venue}', fontsize=12, fontweight='bold', color='#667eea')
            axes[0, 0].grid(True, alpha=0.15, color='white')
            axes[0, 0].set_facecolor('#1a1f3a')
            axes[0, 0].tick_params(colors='white')
            
            # Energy decay
            envelope = np.abs(h) + 1e-10
            energy = 10*np.log10(envelope**2)
            axes[0, 1].plot(t_h, energy, linewidth=2.5, color='#f5576c', alpha=0.85)
            axes[0, 1].set_ylabel('Energy (dB)', fontweight='bold', color='white')
            axes[0, 1].set_title('Energy Decay Curve', fontsize=12, fontweight='bold', color='#667eea')
            axes[0, 1].grid(True, alpha=0.15, color='white')
            axes[0, 1].set_facecolor('#1a1f3a')
            axes[0, 1].tick_params(colors='white')
            
            # Frequency spectrum
            freq_h = fftfreq(len(h), 1/st.session_state.Fs)[:len(h)//2]
            mag_h = np.abs(fft(h))[:len(h)//2]
            axes[1, 0].semilogy(freq_h, mag_h, linewidth=2.5, color='#667eea', alpha=0.85)
            axes[1, 0].set_xlabel('Frequency (Hz)', fontweight='bold', color='white')
            axes[1, 0].set_ylabel('Magnitude', fontweight='bold', color='white')
            axes[1, 0].set_title('Frequency Spectrum', fontsize=12, fontweight='bold', color='#667eea')
            axes[1, 0].grid(True, alpha=0.15, which='both', color='white')
            axes[1, 0].set_xlim([0, 8000])
            axes[1, 0].set_facecolor('#1a1f3a')
            axes[1, 0].tick_params(colors='white')
            
            # Cumulative energy
            cumsum = np.cumsum(h**2) / np.sum(h**2) * 100
            axes[1, 1].plot(t_h, cumsum, linewidth=2.5, color='#f093fb', alpha=0.85)
            axes[1, 1].set_xlabel('Time (s)', fontweight='bold', color='white')
            axes[1, 1].set_ylabel('Cumulative Energy (%)', fontweight='bold', color='white')
            axes[1, 1].set_title('Cumulative Energy Build-Up', fontsize=12, fontweight='bold', color='#667eea')
            axes[1, 1].grid(True, alpha=0.15, color='white')
            axes[1, 1].set_facecolor('#1a1f3a')
            axes[1, 1].tick_params(colors='white')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.divider()
            st.markdown('<div class="section-header">🔊 Audio Playback</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
                    <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">Original</p>
                </div>
                """, unsafe_allow_html=True)
                st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
            with c2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
                    <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">in {selected_venue}</p>
                </div>
                """, unsafe_allow_html=True)
                st.audio(y, sample_rate=st.session_state.Fs)

# ============================================================================
# TAB 4: ADVANCED FILTERS
# ============================================================================

with tabs[3]:
    st.markdown('<div class="section-header">🔧 Advanced Digital Filters</div>', unsafe_allow_html=True)
    st.markdown("*Design professional IIR filters with precise control*")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ftype = st.selectbox("Family", ["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic"], key="ftype_adv")
    with col2:
        subtype = st.selectbox("Type", ["Lowpass", "Highpass", "Bandpass", "Bandstop"], key="subtype_adv")
    with col3:
        order = st.slider("Order", 2, 10, 5, key="order_adv")
    with col4:
        if ftype != "Butterworth":
            ripple = st.slider("Ripple", 0.1, 3.0, 0.5, key="ripple_adv")
    
    st.divider()
    
    if subtype in ["Lowpass", "Highpass"]:
        cutoff = st.slider("Cutoff Frequency (Hz)", 100, 7900, 2000, 50, key="cutoff_adv")
        filter_dict = {"Butterworth": "butterworth", "Chebyshev I": "cheby1", "Chebyshev II": "cheby2", "Elliptic": "ellip"}
        sos = design_digital_filter(filter_dict[ftype], order, cutoff, st.session_state.Fs, subtype.lower())
    else:
        f_low = st.slider("Low Freq (Hz)", 100, 3900, 500, key="flow_adv")
        f_high = st.slider("High Freq (Hz)", f_low+100, 8000, 4000, key="fhigh_adv")
        filter_dict = {"Butterworth": "butterworth", "Chebyshev I": "cheby1", "Chebyshev II": "cheby2", "Elliptic": "ellip"}
        sos = design_digital_filter(filter_dict[ftype], order, (f_low, f_high), st.session_state.Fs, subtype.lower())
    
    if sos is not None:
        st.divider()
        st.markdown('<div class="section-header">📊 Filter Response</div>', unsafe_allow_html=True)
        
        w, mag, phase = compute_freq_response(sos, st.session_state.Fs)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 11))
        fig.patch.set_facecolor('#0a0e27')
        
        # Magnitude response
        axes[0].plot(w, 20*np.log10(mag + 1e-10), linewidth=3, color='#667eea', alpha=0.85)
        axes[0].fill_between(w, 20*np.log10(mag + 1e-10), alpha=0.15, color='#667eea')
        axes[0].set_ylabel('Magnitude (dB)', fontsize=12, fontweight='bold', color='white')
        axes[0].set_title('Magnitude Response', fontsize=13, fontweight='bold', color='#667eea')
        axes[0].grid(True, alpha=0.15, color='white')
        axes[0].axhline(-3, color='#f5576c', linestyle='--', alpha=0.6, linewidth=2, label='-3dB (Cutoff)')
        axes[0].set_xlim([0, 8000])
        axes[0].set_facecolor('#1a1f3a')
        axes[0].tick_params(colors='white', labelsize=10)
        axes[0].legend(fontsize=10, facecolor='#1a1f3a', edgecolor='white', framealpha=0.9)
        
        # Phase response
        axes[1].plot(w, np.degrees(phase), linewidth=3, color='#43e97b', alpha=0.85)
        axes[1].set_ylabel('Phase (degrees)', fontsize=12, fontweight='bold', color='white')
        axes[1].set_title('Phase Response', fontsize=13, fontweight='bold', color='#667eea')
        axes[1].grid(True, alpha=0.15, color='white')
        axes[1].set_xlim([0, 8000])
        axes[1].set_facecolor('#1a1f3a')
        axes[1].tick_params(colors='white', labelsize=10)
        
        # Group delay
        group_delay = -np.diff(np.unwrap(phase)) / np.diff(2*np.pi*w/st.session_state.Fs)
        axes[2].plot(w[:-1], group_delay, linewidth=3, color='#f5576c', alpha=0.85)
        axes[2].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold', color='white')
        axes[2].set_ylabel('Group Delay (samples)', fontsize=12, fontweight='bold', color='white')
        axes[2].set_title('Group Delay', fontsize=13, fontweight='bold', color='#667eea')
        axes[2].grid(True, alpha=0.15, color='white')
        axes[2].set_xlim([0, 8000])
        axes[2].set_facecolor('#1a1f3a')
        axes[2].tick_params(colors='white', labelsize=10)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        if st.button("🎵 APPLY FILTER & ANALYZE", key="apply_adv_filter", type="primary", use_container_width=True):
            filtered = apply_filter(st.session_state.x, sos)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
            with c2:
                st.markdown("**Filtered**")
                st.audio(filtered, sample_rate=st.session_state.Fs)

# ============================================================================
# TAB 5: SIGNAL ANALYSIS
# ============================================================================

with tabs[4]:
    st.markdown('<div class="section-header">📊 Signal Analysis</div>', unsafe_allow_html=True)
    st.markdown("*Advanced time-frequency and spectral analysis*")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    analysis_type = st.selectbox("Analysis Type",
        ["Spectrogram", "Power Spectrum", "Autocorrelation", "Zero Crossings", "MFCC-like"],
        key="analysis_type")
    
    st.divider()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0a0e27')
    ax.set_facecolor('#1a1f3a')
    
    if analysis_type == "Spectrogram":
        f, t, Sxx = signal.spectrogram(st.session_state.x, st.session_state.Fs, nperseg=1024)
        im = ax.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='twilight_shifted')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold', color='white', fontsize=11)
        ax.set_xlabel('Time (s)', fontweight='bold', color='white', fontsize=11)
        ax.set_title('Spectrogram - Time-Frequency Analysis', fontsize=13, fontweight='bold', color='#667eea')
        ax.set_ylim([0, 8000])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontweight='bold', fontsize=10)
        cbar.ax.tick_params(colors='white')
    
    elif analysis_type == "Power Spectrum":
        freqs, psd = signal.welch(st.session_state.x, st.session_state.Fs, nperseg=1024)
        ax.semilogy(freqs, psd, linewidth=2.5, color='#667eea', alpha=0.85)
        ax.fill_between(freqs, psd, alpha=0.15, color='#667eea')
        ax.set_xlabel('Frequency (Hz)', fontweight='bold', color='white', fontsize=11)
        ax.set_ylabel('Power/Freq', fontweight='bold', color='white', fontsize=11)
        ax.set_title('Power Spectral Density (Welch)', fontsize=13, fontweight='bold', color='#667eea')
        ax.grid(True, alpha=0.15, which='both', color='white')
        ax.set_xlim([0, 8000])
    
    elif analysis_type == "Autocorrelation":
        acf = np.correlate(st.session_state.x - np.mean(st.session_state.x), 
                          st.session_state.x - np.mean(st.session_state.x), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        lags = np.arange(len(acf)) / st.session_state.Fs
        ax.plot(lags[:int(0.5*st.session_state.Fs)], acf[:int(0.5*st.session_state.Fs)], 
               linewidth=2.5, color='#43e97b', alpha=0.85)
        ax.axhline(0, color='white', linestyle='--', alpha=0.3)
        ax.set_xlabel('Lag (s)', fontweight='bold', color='white', fontsize=11)
        ax.set_ylabel('Autocorrelation', fontweight='bold', color='white', fontsize=11)
        ax.set_title('Autocorrelation Function', fontsize=13, fontweight='bold', color='#667eea')
        ax.grid(True, alpha=0.15, color='white')
    
    elif analysis_type == "Zero Crossings":
        zc = np.where(np.diff(np.sign(st.session_state.x)))[0]
        ax.plot(st.session_state.x[:5000], linewidth=1.5, color='#667eea', label='Signal', alpha=0.8)
        ax.scatter(zc[zc < 5000], st.session_state.x[zc[zc < 5000]], color='#f5576c', s=60, zorder=5, label='Zero Crossings')
        ax.set_xlabel('Sample', fontweight='bold', color='white', fontsize=11)
        ax.set_ylabel('Amplitude', fontweight='bold', color='white', fontsize=11)
        ax.set_title(f'Zero Crossings Analysis ({len(zc)} total crossings)', fontsize=13, fontweight='bold', color='#667eea')
        ax.grid(True, alpha=0.15, color='white')
        ax.legend(fontsize=10, facecolor='#1a1f3a', edgecolor='white', framealpha=0.9)
    
    else:  # MFCC-like
        ax.plot(st.session_state.x[:4000], linewidth=2, color='#667eea', alpha=0.85)
        ax.fill_between(range(len(st.session_state.x[:4000])), st.session_state.x[:4000], alpha=0.15, color='#667eea')
        ax.set_xlabel('Sample', fontweight='bold', color='white', fontsize=11)
        ax.set_ylabel('Amplitude', fontweight='bold', color='white', fontsize=11)
        ax.set_title('Signal Waveform - Time Domain Detail', fontsize=13, fontweight='bold', color='#667eea')
        ax.grid(True, alpha=0.15, color='white')
    
    ax.tick_params(colors='white', labelsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# TAB 6: EFFECTS LAB
# ============================================================================

with tabs[5]:
    st.markdown('<div class="section-header">✨ Effects Laboratory</div>', unsafe_allow_html=True)
    st.markdown("*Creative audio effects and sound design*")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    effect = st.selectbox("Effect Type",
        ["Vibrato", "Tremolo", "Distortion", "Echo", "Reverb", "Chorus"],
        key="effect_type")
    
    st.divider()
    
    if effect == "Vibrato":
        rate = st.slider("Rate (Hz)", 2.0, 15.0, 5.0)
        depth = st.slider("Depth", 0.01, 0.5, 0.1)
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        mod = 1 + depth * np.sin(2*np.pi*rate*t)
        processed = st.session_state.x * mod
        effect_desc = f"Vibrato at {rate:.1f}Hz with {depth:.2f} depth"
    
    elif effect == "Tremolo":
        rate = st.slider("Rate (Hz)", 2.0, 15.0, 5.0)
        depth = st.slider("Depth", 0.1, 1.0, 0.5)
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        mod = 1 - depth * (1 + np.sin(2*np.pi*rate*t)) / 2
        processed = st.session_state.x * mod
        effect_desc = f"Tremolo at {rate:.1f}Hz with {depth:.2f} depth"
    
    elif effect == "Distortion":
        gain = st.slider("Gain", 1.0, 20.0, 5.0)
        processed = np.tanh(st.session_state.x * gain)
        effect_desc = f"Distortion with {gain:.1f}x gain"
    
    elif effect == "Echo":
        delay = st.slider("Delay (sec)", 0.1, 1.0, 0.3)
        feedback = st.slider("Feedback", 0.0, 0.95, 0.5)
        delay_samples = int(delay * st.session_state.Fs)
        processed = np.zeros(len(st.session_state.x) + delay_samples * 3)
        processed[:len(st.session_state.x)] = st.session_state.x
        for i in range(3):
            pos = (i+1) * delay_samples
            processed[pos:pos+len(st.session_state.x)] += (feedback**(i+1)) * st.session_state.x
        processed = processed[:len(st.session_state.x) + delay_samples*3]
        effect_desc = f"Echo with {delay:.2f}s delay and {feedback:.2f} feedback"
    
    elif effect == "Reverb":
        decay = st.slider("Decay Time", 1.0, 5.0, 2.5)
        h_rev = generate_acoustic_ir("Concert Hall Acoustics", st.session_state.Fs, decay/10)
        processed = signal.convolve(st.session_state.x, h_rev, mode='same')
        processed = processed / np.max(np.abs(processed)) * 0.95
        effect_desc = f"Reverb with {decay:.1f}s decay"
    
    else:  # Chorus
        rate = st.slider("Rate (Hz)", 0.5, 3.0, 1.5)
        depth = st.slider("Depth (ms)", 5, 30, 15)
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        delay_samples = int((depth / 1000) * st.session_state.Fs)
        modulation = 1 + 0.5 * np.sin(2*np.pi*rate*t)
        processed = st.session_state.x.copy()
        delayed = np.zeros_like(st.session_state.x)
        for i in range(len(st.session_state.x)):
            delay_idx = int(i - delay_samples * modulation[i])
            if delay_idx >= 0:
                delayed[i] = st.session_state.x[delay_idx]
        processed = (st.session_state.x + 0.5 * delayed) / 1.5
        effect_desc = f"Chorus at {rate:.1f}Hz with {depth:.0f}ms depth"
    
    if st.button("🎵 APPLY EFFECT", key="apply_effect", type="primary", use_container_width=True):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor('#0a0e27')
        
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        t_proc = np.arange(len(processed)) / st.session_state.Fs
        
        # Time domain - original
        axes[0, 0].plot(t[:min(8000, len(t))], st.session_state.x[:min(8000, len(st.session_state.x))], 
                       linewidth=2, color='#667eea', alpha=0.85)
        axes[0, 0].fill_between(t[:min(8000, len(t))], st.session_state.x[:min(8000, len(st.session_state.x))], 
                               alpha=0.15, color='#667eea')
        axes[0, 0].set_title('Original Signal', fontsize=12, fontweight='bold', color='#667eea')
        axes[0, 0].set_ylabel('Amplitude', fontsize=11, fontweight='bold', color='white')
        axes[0, 0].grid(True, alpha=0.15, color='white')
        axes[0, 0].set_facecolor('#1a1f3a')
        axes[0, 0].tick_params(colors='white')
        
        # Time domain - processed
        axes[0, 1].plot(t_proc[:min(8000, len(t_proc))], processed[:min(8000, len(processed))], 
                       linewidth=2, color='#f5576c', alpha=0.85)
        axes[0, 1].fill_between(t_proc[:min(8000, len(t_proc))], processed[:min(8000, len(processed))], 
                               alpha=0.15, color='#f5576c')
        axes[0, 1].set_title(f'{effect} Effect (Time Domain)', fontsize=12, fontweight='bold', color='#f5576c')
        axes[0, 1].set_ylabel('Amplitude', fontsize=11, fontweight='bold', color='white')
        axes[0, 1].grid(True, alpha=0.15, color='white')
        axes[0, 1].set_facecolor('#1a1f3a')
        axes[0, 1].tick_params(colors='white')
        
        # Frequency domain
        freq_orig = fftfreq(len(st.session_state.x), 1/st.session_state.Fs)[:len(st.session_state.x)//2]
        mag_orig = np.abs(fft(st.session_state.x))[:len(st.session_state.x)//2]
        freq_proc = fftfreq(len(processed), 1/st.session_state.Fs)[:len(processed)//2]
        mag_proc = np.abs(fft(processed))[:len(processed)//2]
        
        axes[1, 0].semilogy(freq_orig, mag_orig, linewidth=2.5, color='#667eea', alpha=0.85, label='Original')
        axes[1, 0].semilogy(freq_proc, mag_proc, linewidth=2.5, color='#f5576c', alpha=0.85, label=effect)
        axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', color='white')
        axes[1, 0].set_ylabel('Magnitude', fontsize=11, fontweight='bold', color='white')
        axes[1, 0].set_title('Frequency Spectrum Comparison', fontsize=12, fontweight='bold', color='#667eea')
        axes[1, 0].grid(True, alpha=0.15, which='both', color='white')
        axes[1, 0].set_xlim([0, 8000])
        axes[1, 0].set_facecolor('#1a1f3a')
        axes[1, 0].tick_params(colors='white')
        axes[1, 0].legend(fontsize=10, facecolor='#1a1f3a', edgecolor='white', framealpha=0.9)
        
        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(processed, st.session_state.Fs, nperseg=1024)
        im = axes[1, 1].pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='twilight_shifted')
        axes[1, 1].set_ylabel('Frequency (Hz)', fontsize=11, fontweight='bold', color='white')
        axes[1, 1].set_xlabel('Time (s)', fontsize=11, fontweight='bold', color='white')
        axes[1, 1].set_title('Effect Spectrogram', fontsize=12, fontweight='bold', color='#667eea')
        axes[1, 1].set_ylim([0, 8000])
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Power (dB)', fontweight='bold', fontsize=9)
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        st.divider()
        st.markdown('<div class="section-header">🔊 Audio Playback</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;
                        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">Original</p>
            </div>
            """, unsafe_allow_html=True)
            st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
        with c2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;
                        box-shadow: 0 8px 25px rgba(245, 87, 108, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">{effect} Effect</p>
            </div>
            """, unsafe_allow_html=True)
            st.audio(processed, sample_rate=st.session_state.Fs)

st.divider()
st.markdown("""
<div class="info-box">
    <h4 style="margin-top: 0;">💡 About LSI Acoustic Studio PRO</h4>
    <p style="margin-bottom: 0;">This advanced DSP platform combines real-time signal processing, acoustic simulation of famous world venues, 
    professional music filters, and creative effects in one comprehensive tool. All computations use NumPy/SciPy for high performance and scientific accuracy.</p>
</div>
""", unsafe_allow_html=True)
