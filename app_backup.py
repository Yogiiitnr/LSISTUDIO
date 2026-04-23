# coding: utf-8
"""
LSI Acoustic Studio - Professional Interactive DSP Platform
Advanced educational tool with enhanced UI/UX and complete functionality
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import butter, cheby1, cheby2, ellip, sosfilt, sosfreqz
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LSI Acoustic Studio Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ADVANCED STYLING - RICH COLOR PALETTE
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 0;
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }
    
    .main > div:first-child {
        padding: 2rem;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        color: white;
    }
    
    .header-container h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-container p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Card Styling */
    .card-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .card-header {
        color: #667eea;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-card.accent1 {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-card.accent2 {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-card.accent3 {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    
    .metric-card.accent4 {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"] {
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 2px solid #667eea !important;
    }
    
    /* Input Elements */
    .stSlider, .stSelectbox, .stRadio {
        background: transparent;
    }
    
    /* Info/Warning Boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(67, 233, 123, 0.1) 0%, rgba(56, 249, 215, 0.1) 100%);
        border-left: 5px solid #43e97b;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #667eea;
        font-weight: 700;
    }
    
    h2 {
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.75rem;
        margin-top: 2rem;
    }
    
    p {
        color: #e0e0e0;
        line-height: 1.6;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(102, 126, 234, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
    }
</style>
""", unsafe_allow_html=True)

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
        except Exception as e:
            # Don't show error for initial state - just return None
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
        fundamental = 100
        harmonics = np.sin(2 * np.pi * fundamental * t)
        for h in [2, 3, 4]:
            harmonics += 0.3 / h * np.sin(2 * np.pi * fundamental * h * t)
        return harmonics / 3
    
    elif signal_type == 'White Noise':
        return np.random.normal(0, 0.3, len(t))
    
    elif signal_type == 'Impulse':
        sig = np.zeros_like(t)
        sig[int(0.1*sr)] = 1.0
        return sig
    
    return 0.5 * np.sin(2 * np.pi * 440 * t)

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
    """Apply filter"""
    return sosfilt(sos, sig)

def compute_freq_response(sos, sr):
    """Compute filter frequency response"""
    w = np.logspace(0, np.log10(sr/2), 1024)
    w_rad = 2 * np.pi * w / sr
    w_rad, h = sosfreqz(sos, w_rad)
    return w_rad * sr / (2 * np.pi), np.abs(h), np.angle(h)

def generate_room_ir(room_type, sr, duration=1.0):
    """Generate room impulse response"""
    num_samples = int(duration * sr)
    h = np.zeros(num_samples)
    
    configs = {
        'Small Room': {'decay': 2.0, 'reflections': 8, 'spacing': 0.02},
        'Concert Hall': {'decay': 6.0, 'reflections': 20, 'spacing': 0.04},
        'Cathedral': {'decay': 10.0, 'reflections': 40, 'spacing': 0.05},
    }
    
    cfg = configs.get(room_type, configs['Concert Hall'])
    h[0] = 1.0
    
    decay_rate = cfg['decay'] / 1000
    spacing_samples = int(cfg['spacing'] * sr)
    
    for i in range(cfg['reflections']):
        delay = (i + 1) * spacing_samples
        if delay < len(h):
            h[delay] = np.exp(-decay_rate * delay) * (1 + 0.1 * np.random.random())
    
    max_val = np.max(np.abs(h))
    if max_val > 0:
        h = h / max_val
    
    return h

# ============================================================================
# HEADER
# ============================================================================

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header-container">
    <h1>🎵 LSI Acoustic Studio PRO</h1>
    <p style="font-size: 1.1rem; margin-top: 1rem;">Professional Interactive DSP & Room Acoustics Platform | Signal Processing & Systems Theory</p>
    <p style="font-size: 0.95rem; opacity: 0.9; margin-top: 0.5rem;">Advanced educational tool for real-time audio convolution, filter design, and frequency analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Success and Error Message Styling */
    .stSuccess {
        background-color: rgba(67, 233, 123, 0.15) !important;
        border-left: 5px solid #43e97b !important;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.15) !important;
        border-left: 5px solid #ffc107 !important;
    }
    
    .stError {
        background-color: rgba(245, 87, 108, 0.15) !important;
        border-left: 5px solid #f5576c !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - GLOBAL AUDIO INPUT
# ============================================================================

with st.sidebar:
    st.markdown("# ⚙️ SIGNAL INPUT")
    
    # Initialize session state
    if 'x' not in st.session_state:
        st.session_state.x = None
    if 'Fs' not in st.session_state:
        st.session_state.Fs = None
    
    audio_source = st.radio(
        "Select Audio Source",
        ["🎤 Microphone", "📁 Upload File", "🔧 Generate Test"],
        key="audio_source_sidebar"
    )
    
    if audio_source == "🎤 Microphone":
        x, Fs = record_audio_from_mic()
        if x is not None:
            st.session_state.x = x
            st.session_state.Fs = Fs
            st.success("✅ Microphone recorded")
    
    elif audio_source == "📁 Upload File":
        uploaded = st.file_uploader("Upload .wav", type=["wav"])
        if uploaded:
            x, Fs = load_audio_from_file(uploaded)
            if x is not None:
                st.session_state.x = x
                st.session_state.Fs = Fs
                st.success("✅ File loaded")
    
    else:
        signal_type = st.selectbox("Test Signal Type",
            ['Speech-like (Chirp + Noise)', 'Pure Tone (440 Hz - A4)', 
             'Harmonic Complex (100 Hz)', 'White Noise', 'Impulse'],
            key="signal_type_select")
        
        if st.button("Generate Signal", width='stretch'):
            x = generate_test_signal(signal_type)
            Fs = 16000
            x, Fs = preprocess_audio(x, Fs)
            st.session_state.x = x
            st.session_state.Fs = Fs
            st.success("✅ Test signal generated")
    
    # Show loaded signal info
    if st.session_state.x is not None and st.session_state.Fs is not None:
        st.divider()
        st.markdown("### 📊 LOADED SIGNAL")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duration", f"{len(st.session_state.x)/st.session_state.Fs:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{st.session_state.Fs} Hz")
        
        # Play loaded signal
        st.audio(st.session_state.x, sample_rate=st.session_state.Fs)

# ============================================================================
# MAIN TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎛️ Convolution",
    "🔧 Filter Designer",
    "📊 Analysis",
    "🏛️ Room Acoustics",
    "✨ Effects",
    "📚 Theory"
])

# ============================================================================
# TAB 1: CONVOLUTION ENGINE
# ============================================================================

with tab1:
    st.markdown("## Convolution Simulator")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Impulse Response Design")
        
        ir_mode = st.radio("IR Type", ["Echo", "Room Preset"], key="ir_mode_tab1")
        
        if ir_mode == "Echo":
            delay = st.slider("Delay (sec)", 0.1, 1.0, 0.3, 0.05, key="echo_delay_tab1")
            attenuation = st.slider("Attenuation", 0.1, 0.95, 0.5, 0.05, key="echo_atten_tab1")
            num_echoes = st.slider("Echoes", 1, 5, 3, key="num_echoes_tab1")
            
            delay_samples = int(delay * st.session_state.Fs)
            h = np.zeros(delay_samples * (num_echoes + 1))
            h[0] = 1.0
            for i in range(1, num_echoes + 1):
                pos = i * delay_samples
                if pos < len(h):
                    h[pos] = attenuation ** i
        else:
            room = st.selectbox("Room Type", ["Small Room", "Concert Hall", "Cathedral"], key="room_tab1")
            h = generate_room_ir(room, st.session_state.Fs, duration=1.0)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'<div class="metric-card"><div class="metric-label">IR Length</div><div class="metric-value">{len(h)/st.session_state.Fs:.3f}s</div></div>', unsafe_allow_html=True)
        with col_b:
            st.markdown(f'<div class="metric-card accent1"><div class="metric-label">Samples</div><div class="metric-value">{len(h)}</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ⚙️ Processing Options")
        
        apply_prefilter = st.checkbox("Apply Pre-Filter", key="apply_prefilter_tab1")
        
        if apply_prefilter:
            filter_type = st.selectbox("Filter", ["Butterworth", "Chebyshev I"], key="filter_tab1")
            order = st.slider("Order", 2, 8, 4, key="order_tab1")
            cutoff = st.slider("Cutoff (Hz)", 100, 8000, 2000, key="cutoff_tab1")
            
            filter_dict = {"Butterworth": "butterworth", "Chebyshev I": "cheby1"}
            sos = design_digital_filter(filter_dict[filter_type], order, cutoff, st.session_state.Fs)
            x_proc = apply_filter(st.session_state.x, sos) if sos else st.session_state.x
        else:
            x_proc = st.session_state.x.copy()
    
    st.divider()
    
    if st.button("🔊 COMPUTE CONVOLUTION", width='stretch', type="primary"):
        if st.session_state.x is None:
            st.error("❌ Load audio first!")
        else:
            y = signal.convolve(x_proc, h, mode='full')
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / (max_val * 1.2)
            
            t_x = np.arange(len(x_proc)) / st.session_state.Fs
            t_h = np.arange(len(h)) / st.session_state.Fs
            t_y = np.arange(len(y)) / st.session_state.Fs
            
            # Visualization
            fig = plt.figure(figsize=(16, 12))
            fig.patch.set_facecolor('#0f0f1e')
            gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(t_x, x_proc, linewidth=2.5, color='#667eea', alpha=0.90, label='Input Signal')
            ax1.fill_between(t_x, x_proc, alpha=0.20, color='#667eea')
            ax1.axhline(y=0, color='white', linewidth=0.8, alpha=0.3, linestyle='--')
            ax1.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
            ax1.set_title('Input Signal x[n] - Time Domain', fontsize=13, fontweight='bold', color='#667eea')
            ax1.grid(True, alpha=0.20, linestyle='--', color='white')
            ax1.set_facecolor('#1a1a2e')
            ax1.tick_params(colors='white', labelsize=10)
            ax1.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e', edgecolor='white')
            ax1.set_xlim([0, max(t_x)])
            
            ax2 = fig.add_subplot(gs[1, 0])
            markerline, stemlines, baseline = ax2.stem(t_h[:min(500, len(t_h))], h[:min(500, len(h))], basefmt=' ')
            stemlines.set_color('#43e97b')
            stemlines.set_linewidth(2.2)
            markerline.set_color('#43e97b')
            markerline.set_markersize(8)
            baseline.set_color('white')
            baseline.set_linewidth(0.8)
            baseline.set_alpha(0.3)
            ax2.axhline(y=0, color='white', linewidth=0.8, alpha=0.3, linestyle='--')
            ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
            ax2.set_title(f'Impulse Response h[n] - {ir_mode} Mode', fontsize=13, fontweight='bold', color='#43e97b')
            ax2.grid(True, alpha=0.20, color='white')
            ax2.set_facecolor('#1a1a2e')
            ax2.tick_params(colors='white', labelsize=10)
            ax2.set_xlim([0, min(t_h[-1], 0.5)])
            
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(t_y, y, linewidth=2.5, color='#f5576c', alpha=0.90, label='Output Signal')
            ax3.fill_between(t_y, y, alpha=0.20, color='#f5576c')
            ax3.axhline(y=0, color='white', linewidth=0.8, alpha=0.3, linestyle='--')
            ax3.set_ylabel('Amplitude', fontsize=12, fontweight='bold', color='white')
            ax3.set_title('Output Signal y[n] = x[n] * h[n]', fontsize=13, fontweight='bold', color='#f5576c')
            ax3.grid(True, alpha=0.20, color='white')
            ax3.set_xlim([0, min(t_y[-1], 5.0)])
            ax3.set_facecolor('#1a1a2e')
            ax3.tick_params(colors='white', labelsize=10)
            ax3.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e', edgecolor='white')
            
            ax4 = fig.add_subplot(gs[2, :])
            freq_x, mag_x = fftfreq(len(x_proc), 1/st.session_state.Fs)[:len(x_proc)//2], np.abs(fft(x_proc))[:len(x_proc)//2]
            freq_y, mag_y = fftfreq(len(y), 1/st.session_state.Fs)[:len(y)//2], np.abs(fft(y))[:len(y)//2]
            
            mag_x_db = 20*np.log10(mag_x + 1e-10)
            mag_y_db = 20*np.log10(mag_y + 1e-10)
            
            ax4.semilogy(freq_x, mag_x_db, label='Input Spectrum', linewidth=2.5, color='#667eea', alpha=0.85)
            ax4.semilogy(freq_y, mag_y_db, label='Output Spectrum (Convolved)', linewidth=2.5, color='#f5576c', alpha=0.85)
            ax4.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold', color='white')
            ax4.set_ylabel('Magnitude (dB)', fontsize=12, fontweight='bold', color='white')
            ax4.set_title('Frequency Domain Analysis - Comb Filter Effect (Peaks & Nulls)', fontsize=13, fontweight='bold', color='#667eea')
            ax4.grid(True, alpha=0.20, which='both', color='white', linestyle='--')
            ax4.set_xlim([0, 8000])
            ax4.set_facecolor('#1a1a2e')
            ax4.tick_params(colors='white', labelsize=10)
            ax4.legend(fontsize=11, loc='upper right', facecolor='#1a1a2e', edgecolor='white', framealpha=0.95)
            
            st.pyplot(fig, width='stretch')
            
            st.divider()
            col_play = st.columns(1)[0]
            with col_play:
                st.markdown("### 🔊 Audio Playback & Comparison")
                
                audio_col1, audio_col2, audio_col3 = st.columns(3)
                with audio_col1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                        <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">🎵 Original Signal</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.audio(x_proc, sample_rate=st.session_state.Fs)
                    st.caption(f"Duration: {len(x_proc)/st.session_state.Fs:.2f}s | Sample Rate: {st.session_state.Fs}Hz")
                with audio_col2:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                        <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">📢 Convolved Output</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.audio(y, sample_rate=st.session_state.Fs)
                    st.caption(f"Duration: {len(y)/st.session_state.Fs:.2f}s | Normalized")
                with audio_col3:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                        <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0;">🎯 Impulse Response</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.audio(h, sample_rate=st.session_state.Fs)
                    st.caption(f"Duration: {len(h)/st.session_state.Fs:.2f}s")
            
            st.divider()
            st.markdown("### 📊 Processing Metrics & Analysis")
            
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
            
            # Additional analysis metrics
            st.divider()
            st.markdown("### 🔬 Detailed Analysis")
            
            col_analysis1, col_analysis2 = st.columns(2)
            with col_analysis1:
                st.markdown("**Time Domain Properties**")
                analysis_data1 = {
                    "Metric": ["Input Duration", "Output Duration", "Expansion Factor", "Input RMS", "Output RMS"],
                    "Value": [
                        f"{len(x_proc)/st.session_state.Fs:.3f}s",
                        f"{len(y)/st.session_state.Fs:.3f}s",
                        f"{len(y)/len(x_proc):.2f}x",
                        f"{np.sqrt(np.mean(x_proc**2)):.4f}",
                        f"{np.sqrt(np.mean(y**2)):.4f}"
                    ]
                }
                analysis_df1 = st.dataframe(analysis_data1, use_container_width=True, hide_index=True)
            
            with col_analysis2:
                st.markdown("**Frequency Domain Properties**")
                analysis_data2 = {
                    "Metric": ["Input Peak Freq", "Output Peak Freq", "Max Input Magnitude", "Max Output Magnitude", "Magnification"],
                    "Value": [
                        f"{freq_x[np.argmax(mag_x)]:.1f} Hz",
                        f"{freq_y[np.argmax(mag_y)]:.1f} Hz",
                        f"{np.max(mag_x):.2e}",
                        f"{np.max(mag_y):.2e}",
                        f"{np.max(mag_y)/np.max(mag_x):.2f}x"
                    ]
                }
                analysis_df2 = st.dataframe(analysis_data2, use_container_width=True, hide_index=True)
            
            st.divider()
            st.markdown("""
            **💡 About the Comb Filter Effect:**
            - The output spectrum shows peaks and nulls due to constructive and destructive interference from echoes
            - Echo delay determines the frequency spacing of peaks: $\\Delta f = F_s / D$ (where D = delay in samples)
            - Higher attenuation factor reduces the prominence of echoes and nulls in the frequency response
            """)

# ============================================================================
# TAB 2: FILTER DESIGNER
# ============================================================================

with tab2:
    st.markdown("## Digital Filter Designer")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        ftype = st.selectbox("Family", ["Butterworth", "Chebyshev I", "Chebyshev II", "Elliptic"], key="ftype_tab2")
    with col2:
        subtype = st.selectbox("Type", ["Lowpass", "Highpass", "Bandpass", "Bandstop"], key="subtype_tab2")
    with col3:
        order = st.slider("Order", 2, 10, 5, key="order_tab2")
    with col4:
        if ftype != "Butterworth":
            ripple = st.slider("Ripple", 0.1, 3.0, 0.5, key="ripple_tab2")
    
    st.divider()
    
    if subtype in ["Lowpass", "Highpass"]:
        cutoff = st.slider("Cutoff Frequency (Hz)", 100, 7900, 2000, 50, key="cutoff_tab2")
        filter_dict = {"Butterworth": "butterworth", "Chebyshev I": "cheby1", "Chebyshev II": "cheby2", "Elliptic": "ellip"}
        sos = design_digital_filter(filter_dict[ftype], order, cutoff, st.session_state.Fs, subtype.lower())
    else:
        f_low = st.slider("Low Freq (Hz)", 100, 3900, 500, key="flow_tab2")
        f_high = st.slider("High Freq (Hz)", f_low+100, 8000, 4000, key="fhigh_tab2")
        filter_dict = {"Butterworth": "butterworth", "Chebyshev I": "cheby1", "Chebyshev II": "cheby2", "Elliptic": "ellip"}
        sos = design_digital_filter(filter_dict[ftype], order, (f_low, f_high), st.session_state.Fs, subtype.lower())
    
    if sos is not None:
        st.divider()
        st.markdown("### 📊 Frequency Response")
        
        w, mag, phase = compute_freq_response(sos, st.session_state.Fs)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.patch.set_facecolor('#0f0f1e')
        
        axes[0].plot(w, 20*np.log10(mag + 1e-10), linewidth=2.5, color='#667eea')
        axes[0].fill_between(w, 20*np.log10(mag + 1e-10), alpha=0.15, color='#667eea')
        axes[0].set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold', color='white')
        axes[0].set_title('Magnitude Response', fontsize=12, fontweight='bold', color='#667eea')
        axes[0].grid(True, alpha=0.15, color='white')
        axes[0].axhline(-3, color='#f5576c', linestyle='--', alpha=0.5, label='-3dB')
        axes[0].set_xlim([0, 8000])
        axes[0].set_facecolor('#1a1a2e')
        axes[0].tick_params(colors='white')
        axes[0].legend()
        
        axes[1].plot(w, phase, linewidth=2.5, color='#43e97b')
        axes[1].set_ylabel('Phase (rad)', fontsize=11, fontweight='bold', color='white')
        axes[1].set_title('Phase Response', fontsize=12, fontweight='bold', color='#667eea')
        axes[1].grid(True, alpha=0.15, color='white')
        axes[1].set_xlim([0, 8000])
        axes[1].set_facecolor('#1a1a2e')
        axes[1].tick_params(colors='white')
        
        # Group delay
        group_delay = -np.diff(np.unwrap(phase)) / np.diff(2*np.pi*w/st.session_state.Fs)
        axes[2].plot(w[:-1], group_delay, linewidth=2.5, color='#f093fb')
        axes[2].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold', color='white')
        axes[2].set_ylabel('Group Delay (samples)', fontsize=11, fontweight='bold', color='white')
        axes[2].set_title('Group Delay', fontsize=12, fontweight='bold', color='#667eea')
        axes[2].grid(True, alpha=0.15, color='white')
        axes[2].set_xlim([0, 8000])
        axes[2].set_facecolor('#1a1a2e')
        axes[2].tick_params(colors='white')
        
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        
        st.divider()
        st.markdown("### 🎵 Apply to Signal")
        
        if st.session_state.x is not None and st.button("Apply Filter", width='stretch', key="apply_filter_tab2"):
            filtered = apply_filter(st.session_state.x, sos)
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Original**")
                st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
            with c2:
                st.write("**Filtered**")
                st.audio(filtered, sample_rate=st.session_state.Fs)

# ============================================================================
# TAB 3: ADVANCED ANALYSIS
# ============================================================================

with tab3:
    st.markdown("## Signal Analysis")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    analysis = st.selectbox("Analysis Type",
        ["Spectrogram", "Power Spectrum", "Autocorrelation", "Zero Crossings"],
        key="analysis_tab3")
    
    st.divider()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1e')
    ax.set_facecolor('#1a1a2e')
    
    if analysis == "Spectrogram":
        f, t, Sxx = signal.spectrogram(st.session_state.x, st.session_state.Fs, nperseg=1024)
        im = ax.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold', color='white')
        ax.set_xlabel('Time (s)', fontweight='bold', color='white')
        ax.set_title('Spectrogram', fontsize=12, fontweight='bold', color='#667eea')
        ax.set_ylim([0, 8000])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontweight='bold')
    
    elif analysis == "Power Spectrum":
        freqs, psd = signal.welch(st.session_state.x, st.session_state.Fs, nperseg=1024)
        ax.semilogy(freqs, psd, linewidth=2.5, color='#667eea')
        ax.fill_between(freqs, psd, alpha=0.15, color='#667eea')
        ax.set_xlabel('Frequency (Hz)', fontweight='bold', color='white')
        ax.set_ylabel('Power/Freq', fontweight='bold', color='white')
        ax.set_title('Power Spectral Density', fontsize=12, fontweight='bold', color='#667eea')
        ax.grid(True, alpha=0.15, which='both', color='white')
        ax.set_xlim([0, 8000])
    
    elif analysis == "Autocorrelation":
        acf = np.correlate(st.session_state.x - np.mean(st.session_state.x), st.session_state.x - np.mean(st.session_state.x), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        lags = np.arange(len(acf)) / st.session_state.Fs
        ax.plot(lags[:int(0.5*st.session_state.Fs)], acf[:int(0.5*st.session_state.Fs)], linewidth=2.5, color='#43e97b')
        ax.set_xlabel('Lag (s)', fontweight='bold', color='white')
        ax.set_ylabel('Autocorrelation', fontweight='bold', color='white')
        ax.set_title('Autocorrelation', fontsize=12, fontweight='bold', color='#667eea')
        ax.grid(True, alpha=0.15, color='white')
    
    else:  # Zero Crossings
        zc = np.where(np.diff(np.sign(st.session_state.x)))[0]
        ax.plot(st.session_state.x[:5000], linewidth=1.5, color='#667eea', label='Signal')
        ax.scatter(zc[zc < 5000], st.session_state.x[zc[zc < 5000]], color='#f5576c', s=50, zorder=5, label='Zero Crossings')
        ax.set_xlabel('Sample', fontweight='bold', color='white')
        ax.set_ylabel('Amplitude', fontweight='bold', color='white')
        ax.set_title(f'Zero Crossings ({len(zc)} total)', fontsize=12, fontweight='bold', color='#667eea')
        ax.grid(True, alpha=0.15, color='white')
        ax.legend()
    
    ax.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig, width='stretch')

# ============================================================================
# TAB 4: ROOM ACOUSTICS
# ============================================================================

with tab4:
    st.markdown("## Room Acoustics Simulator")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    room = st.selectbox("Room Type", ["Small Room", "Concert Hall", "Cathedral"], key="room_tab4")
    
    st.divider()
    
    if st.session_state.x is not None and st.button("Simulate Acoustics", width='stretch', key="sim_room_tab4"):
        h = generate_room_ir(room, st.session_state.Fs)
        y = signal.convolve(st.session_state.x, h, mode='full')
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / (max_val * 1.2)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#0f0f1e')
        
        t_h = np.arange(len(h)) / st.session_state.Fs
        
        axes[0, 0].plot(t_h, h, linewidth=1.5, color='#667eea')
        axes[0, 0].fill_between(t_h, h, alpha=0.15, color='#667eea')
        axes[0, 0].set_ylabel('Amplitude', fontweight='bold', color='white')
        axes[0, 0].set_title('Impulse Response', fontsize=11, fontweight='bold', color='#667eea')
        axes[0, 0].grid(True, alpha=0.15, color='white')
        axes[0, 0].set_facecolor('#1a1a2e')
        axes[0, 0].tick_params(colors='white')
        
        envelope = np.abs(h)
        envelope = np.maximum(envelope, 1e-10)
        energy = 10*np.log10(envelope**2)
        axes[0, 1].plot(t_h, energy, linewidth=2, color='#43e97b')
        axes[0, 1].set_ylabel('Energy (dB)', fontweight='bold', color='white')
        axes[0, 1].set_title('Energy Decay', fontsize=11, fontweight='bold', color='#667eea')
        axes[0, 1].grid(True, alpha=0.15, color='white')
        axes[0, 1].set_facecolor('#1a1a2e')
        axes[0, 1].tick_params(colors='white')
        
        freq_h = fftfreq(len(h), 1/st.session_state.Fs)[:len(h)//2]
        mag_h = np.abs(fft(h))[:len(h)//2]
        axes[1, 0].semilogy(freq_h, mag_h, linewidth=2, color='#f5576c')
        axes[1, 0].set_xlabel('Frequency (Hz)', fontweight='bold', color='white')
        axes[1, 0].set_ylabel('Magnitude', fontweight='bold', color='white')
        axes[1, 0].set_title('Spectrum', fontsize=11, fontweight='bold', color='#667eea')
        axes[1, 0].grid(True, alpha=0.15, which='both', color='white')
        axes[1, 0].set_xlim([0, 8000])
        axes[1, 0].set_facecolor('#1a1a2e')
        axes[1, 0].tick_params(colors='white')
        
        cumsum = np.cumsum(h**2) / np.sum(h**2) * 100
        axes[1, 1].plot(t_h, cumsum, linewidth=2, color='#f093fb')
        axes[1, 1].set_xlabel('Time (s)', fontweight='bold', color='white')
        axes[1, 1].set_ylabel('Cumulative Energy (%)', fontweight='bold', color='white')
        axes[1, 1].set_title('Cumulative Energy', fontsize=11, fontweight='bold', color='#667eea')
        axes[1, 1].grid(True, alpha=0.15, color='white')
        axes[1, 1].set_facecolor('#1a1a2e')
        axes[1, 1].tick_params(colors='white')
        
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        
        st.divider()
        st.markdown("### 🎵 Audio Playback")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Original**")
            st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
        with c2:
            st.write(f"**In {room}**")
            st.audio(y, sample_rate=st.session_state.Fs)

# ============================================================================
# TAB 5: EFFECTS LAB
# ============================================================================

with tab5:
    st.markdown("## Audio Effects Laboratory")
    
    if st.session_state.x is None or st.session_state.Fs is None:
        st.warning("⚠️ Load an audio signal from the sidebar first!")
        st.stop()
    
    effect = st.selectbox("Effect", ["Vibrato", "Tremolo", "Distortion", "Echo"], key="effect_tab5")
    
    st.divider()
    
    if effect == "Vibrato":
        rate = st.slider("Rate (Hz)", 2, 15, 5, key="vib_rate_tab5")
        depth = st.slider("Depth", 0.01, 0.5, 0.1, key="vib_depth_tab5")
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        mod = 1 + depth * np.sin(2*np.pi*rate*t)
        processed = st.session_state.x * mod
    
    elif effect == "Tremolo":
        rate = st.slider("Rate (Hz)", 2, 15, 5, key="trem_rate_tab5")
        depth = st.slider("Depth", 0.1, 1.0, 0.5, key="trem_depth_tab5")
        t = np.arange(len(st.session_state.x)) / st.session_state.Fs
        mod = 1 - depth * (1 + np.sin(2*np.pi*rate*t)) / 2
        processed = st.session_state.x * mod
    
    elif effect == "Distortion":
        gain = st.slider("Gain", 1, 20, 5, key="dist_gain_tab5")
        processed = np.tanh(st.session_state.x * gain)
    
    else:  # Echo
        delay = st.slider("Delay (sec)", 0.1, 1.0, 0.3, key="echo_delay_tab5")
        feedback = st.slider("Feedback", 0.0, 0.95, 0.5, key="echo_fb_tab5")
        
        delay_samples = int(delay * st.session_state.Fs)
        processed = np.zeros(len(st.session_state.x) + delay_samples * 3)
        processed[:len(st.session_state.x)] = st.session_state.x
        
        for i in range(3):
            pos = (i+1) * delay_samples
            processed[pos:pos+len(st.session_state.x)] += (feedback**(i+1)) * st.session_state.x
        
        processed = processed[:len(st.session_state.x) + delay_samples*3]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor('#0f0f1e')
    
    t = np.arange(len(st.session_state.x)) / st.session_state.Fs
    
    axes[0].plot(t[:min(8000, len(t))], st.session_state.x[:min(8000, len(st.session_state.x))], linewidth=1.5, color='#667eea')
    axes[0].set_ylabel('Amplitude', fontweight='bold', color='white')
    axes[0].set_title('Original', fontsize=11, fontweight='bold', color='#667eea')
    axes[0].grid(True, alpha=0.15, color='white')
    axes[0].set_facecolor('#1a1a2e')
    axes[0].tick_params(colors='white')
    
    t_proc = np.arange(len(processed)) / st.session_state.Fs
    axes[1].plot(t_proc[:min(8000, len(t_proc))], processed[:min(8000, len(processed))], linewidth=1.5, color='#f5576c')
    axes[1].set_xlabel('Time (s)', fontweight='bold', color='white')
    axes[1].set_ylabel('Amplitude', fontweight='bold', color='white')
    axes[1].set_title(f'{effect} Applied', fontsize=11, fontweight='bold', color='#667eea')
    axes[1].grid(True, alpha=0.15, color='white')
    axes[1].set_facecolor('#1a1a2e')
    axes[1].tick_params(colors='white')
    
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Original**")
        st.audio(st.session_state.x, sample_rate=st.session_state.Fs)
    with c2:
        st.write(f"**{effect}**")
        st.audio(processed, sample_rate=st.session_state.Fs)

# ============================================================================
# TAB 6: THEORY & LEARNING
# ============================================================================

with tab6:
    st.markdown("## 📚 Signals & Systems Theory")
    
    topic = st.selectbox("Topic",
        ["Sampling Theorem", "Convolution", "Fourier Transform", "Filter Design", "Room Acoustics"],
        key="topic_tab6")
    
    st.divider()
    
    if topic == "Sampling Theorem":
        st.markdown(r"""
        ### Digital Sampling: Continuous to Discrete
        
        **Continuous Signal** → **Sampling at Rate F_s** → **Discrete Signal x[n]**
        
        $$x[n] = x_c(nT) \quad \text{where} \quad T = \frac{1}{F_s}$$
        
        **Nyquist-Shannon Theorem**: To faithfully represent frequencies up to f_max:
        
        $$F_s \geq 2 f_{max}$$
        
        **Nyquist Frequency** = $F_s / 2$ (maximum representable frequency)
        """)
    
    elif topic == "Convolution":
        st.markdown(r"""
        ### Convolution: System Response
        
        $$y[n] = x[n] \circledast h[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]$$
        
        **Steps**: Fold → Shift → Multiply → Sum
        
        **Key Property**: Fully describes LSI system response to any input
        """)
    
    elif topic == "Fourier Transform":
        st.markdown(r"""
        ### Fourier Transform: Time ↔ Frequency
        
        $$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}$$
        
        **Convolution-Multiplication Duality**:
        
        $$y[n] = x[n] \circledast h[n] \quad \Leftrightarrow \quad Y[k] = X[k] \cdot H[k]$$
        
        **Computational Benefit**: FFT reduces O(N²) to O(N log N)
        """)
    
    elif topic == "Filter Design":
        st.markdown(r"""
        ### Digital Filter Design
        
        **IIR vs FIR**:
        - **IIR**: Compact, possible phase distortion (Butterworth, Chebyshev, Elliptic)
        - **FIR**: Linear phase, more coefficients, always stable
        
        **Key Parameters**:
        - Order: Higher = steeper rolloff
        - Cutoff: -3dB point
        - Ripple: Passband/stopband variation
        """)
    
    else:  # Room Acoustics
        st.markdown(r"""
        ### Room Impulse Response
        
        **Structure**: Direct Sound + Early Reflections + Diffuse Tail
        
        **Reverberation Time (RT60)**: Energy decay by 60 dB
        
        **Comb Filter Effect**: Periodic echoes create peaks/nulls
        
        $$h[n] = \delta[n] + \alpha \delta[n-D] + \alpha^2 \delta[n-2D] + \cdots$$
        
        Creates "coloration" - selective frequency boosting/attenuation
        """)

st.divider()
st.markdown("""
<div style='text-align: center; color: #667eea; margin-top: 2rem;'>
    <p style='font-weight: 700; font-size: 1.1rem;'>LSI Acoustic Studio PRO</p>
    <p style='opacity: 0.8;'>Advanced DSP & Room Acoustics Platform</p>
</div>
""", unsafe_allow_html=True)
# coding: utf-8
"""
LSI Acoustic Studio - Advanced Interactive DSP Platform
Professional educational tool for university-level Signals and Systems concepts
Built with modern UI/UX and advanced signal processing capabilities
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import butter, cheby1, cheby2, ellip, sosfilt
import io
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LSI Acoustic Studio Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Professional Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Roboto', sans-serif;
    }
    
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .header-box h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .header-box p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.85;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .tab-label {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem;
        font-weight: 600;
    }
    
    h1, h2, h3 {
        color: #667eea;
        font-weight: 700;
    }
    
    h2 {
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0f7ff 0%, #e8f5ff 100%);
        border-left: 5px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #e8f5e9 100%);
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #fff9e6 100%);
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .button-group {
        display: flex;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .code-block {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        border-left: 4px solid #667eea;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ADVANCED HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def get_filter_design():
    """Get comprehensive filter design methods"""
    return {
        'Butterworth': ('butterworth', 'Maximally flat response'),
        'Chebyshev Type I': ('cheby1', 'Ripple in passband'),
        'Chebyshev Type II': ('cheby2', 'Ripple in stopband'),
        'Elliptic': ('ellip', 'Ripple in both bands')
    }

def design_digital_filter(filter_type, order, cutoff, sr, filter_subtype='low'):
    """Design digital IIR filter"""
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    
    # Ensure cutoff is valid
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    
    if filter_type == 'butterworth':
        sos = butter(order, normalized_cutoff, btype=filter_subtype, output='sos')
    elif filter_type == 'cheby1':
        sos = cheby1(order, 0.1, normalized_cutoff, btype=filter_subtype, output='sos')
    elif filter_type == 'cheby2':
        sos = cheby2(order, 40, normalized_cutoff, btype=filter_subtype, output='sos')
    elif filter_type == 'ellip':
        sos = ellip(order, 0.1, 40, normalized_cutoff, btype=filter_subtype, output='sos')
    
    return sos

def apply_filter(signal_data, sos):
    """Apply digital filter to signal"""
    return sosfilt(sos, signal_data)

def compute_frequency_response(sos, sr, num_points=2048):
    """Compute frequency response of filter"""
    w = np.logspace(0, np.log10(sr/2), num_points)
    w_rad = 2 * np.pi * w / sr
    
    # Compute using digital filter
    from scipy.signal import sosfreqz
    w_rad, h = sosfreqz(sos, w_rad)
    w = w_rad * sr / (2 * np.pi)
    
    magnitude = np.abs(h)
    phase = np.angle(h)
    group_delay = -np.diff(np.unwrap(phase)) / np.diff(w_rad)
    
    return w, magnitude, phase, group_delay

def generate_advanced_room_ir(room_type, sr, duration=1.0, detail_level='high'):
    """Generate advanced room impulse response with multiple reflections"""
    num_samples = int(duration * sr)
    h = np.zeros(num_samples)
    
    room_configs = {
        'Anechoic Chamber': {
            'decay': 0.5,
            'reflections': 3,
            'spacing': 0.02,
            'randomness': 0.05
        },
        'Small Studio': {
            'decay': 3.0,
            'reflections': 8,
            'spacing': 0.03,
            'randomness': 0.15
        },
        'Medium Hall': {
            'decay': 6.0,
            'reflections': 15,
            'spacing': 0.04,
            'randomness': 0.25
        },
        'Large Concert Hall': {
            'decay': 10.0,
            'reflections': 25,
            'spacing': 0.05,
            'randomness': 0.35
        },
        'Cathedral': {
            'decay': 14.0,
            'reflections': 40,
            'spacing': 0.06,
            'randomness': 0.45
        },
        'Reverberant Chamber': {
            'decay': 8.0,
            'reflections': 30,
            'spacing': 0.035,
            'randomness': 0.3
        }
    }
    
    config = room_configs.get(room_type, room_configs['Medium Hall'])
    
    # Primary impulse
    h[0] = 1.0
    
    # Early reflections with detailed structure
    np.random.seed(42)  # Reproducibility
    decay_rate = config['decay'] / 1000
    reflection_spacing = int(config['spacing'] * sr)
    
    for i in range(config['reflections']):
        delay = (i + 1) * reflection_spacing
        if delay < len(h):
            # Exponential decay with random modulation
            amplitude = np.exp(-decay_rate * delay) * (1 + config['randomness'] * (np.random.random() - 0.5))
            h[delay] = max(0, amplitude)
    
    # Diffuse tail (exponential decay)
    tail_start = config['reflections'] * reflection_spacing
    if tail_start < len(h):
        tail_indices = np.arange(tail_start, len(h))
        tail_envelope = np.exp(-decay_rate * tail_indices) * np.random.normal(0.1, 0.05, len(tail_indices))
        h[tail_start:len(h)] += np.maximum(0, tail_envelope)
    
    # Normalize
    max_val = np.max(np.abs(h))
    if max_val > 0:
        h = h / max_val
    
    return h

def compute_reverb_time(h, sr, decay_threshold=-60):
    """Calculate reverberation time (RT60)"""
    envelope = np.abs(h)
    envelope = np.maximum(envelope, 1e-10)
    
    energy_db = 10 * np.log10(envelope ** 2)
    max_energy = np.max(energy_db)
    
    # Find where energy drops to threshold
    threshold = max_energy + decay_threshold
    indices = np.where(energy_db <= threshold)[0]
    
    if len(indices) > 0:
        rt_time = indices[0] / sr
    else:
        rt_time = len(h) / sr
    
    return rt_time

def generate_special_effects(audio, sr, effect_type, **params):
    """Generate special audio effects"""
    if effect_type == 'vibrato':
        rate = params.get('rate', 5)
        depth = params.get('depth', 0.1)
        t = np.arange(len(audio)) / sr
        modulation = 1 + depth * np.sin(2 * np.pi * rate * t)
        return audio * modulation
    
    elif effect_type == 'tremolo':
        rate = params.get('rate', 5)
        depth = params.get('depth', 0.5)
        t = np.arange(len(audio)) / sr
        modulation = 1 - depth * (1 + np.sin(2 * np.pi * rate * t)) / 2
        return audio * modulation
    
    elif effect_type == 'distortion':
        gain = params.get('gain', 5)
        audio_gained = audio * gain
        return np.tanh(audio_gained)
    
    elif effect_type == 'chorus':
        delay_ms = params.get('delay_ms', 50)
        depth = params.get('depth', 0.1)
        delay_samples = int(delay_ms * sr / 1000)
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        return audio + depth * delayed
    
    return audio

def record_audio_from_mic():
    """Record audio from microphone"""
    audio_data = st.audio_input("Record your voice")
    if audio_data is not None:
        try:
            audio_bytes = audio_data.getvalue()
            data, sr = sf.read(io.BytesIO(audio_bytes))
            return data, sr
        except Exception as e:
            st.error(f"Microphone error: {e}")
            return None, None
    return None, None

def load_audio_from_file(uploaded_file):
    """Load audio from file"""
    if uploaded_file is not None:
        try:
            data, sr = sf.read(uploaded_file)
            return data, sr
        except Exception as e:
            st.error(f"File error: {e}")
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
    
    if signal_type == 'Speech-like':
        # Chirp + noise burst
        chirp = signal.chirp(t, 200, duration, 2000, method='linear')
        noise = np.zeros_like(t)
        noise[int(0.5*sr):int(1.0*sr)] = np.random.normal(0, 0.3, int(0.5*sr))
        return 0.7 * chirp + 0.3 * noise
    
    elif signal_type == 'Pure Tone (440 Hz)':
        return 0.5 * np.sin(2 * np.pi * 440 * t)
    
    elif signal_type == 'Harmonic Complex':
        fundamental = 100
        harmonics = np.sin(2 * np.pi * fundamental * t)
        for h in [2, 3, 4]:
            harmonics += 0.5 / h * np.sin(2 * np.pi * fundamental * h * t)
        return harmonics / 4
    
    elif signal_type == 'Noise':
        return np.random.normal(0, 0.5, len(t))
    
    elif signal_type == 'Impulse':
        signal_out = np.zeros_like(t)
        signal_out[int(0.1*sr)] = 1.0
        return signal_out
    
    elif signal_type == 'Step Function':
        signal_out = np.zeros_like(t)
        signal_out[int(0.3*sr):] = 1.0
        return signal_out
    
    return 0.5 * np.sin(2 * np.pi * 440 * t)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header-box">
    <h1>🎵 LSI Acoustic Studio PRO</h1>
    <p>Advanced Interactive Platform for Digital Signal Processing & Room Acoustics</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN NAVIGATION
# ============================================================================

nav_tabs = st.tabs([
    "🎛️ Acoustic Engine",
    "🔧 Filter Designer",
    "📊 Advanced Analysis",
    "🏛️ Room Acoustics",
    "✨ Effects Lab",
    "📚 Theory & Learning"
])

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ INPUT SIGNAL")
    
    audio_source = st.radio("Signal Source", 
        ["Microphone", "Upload File", "Generate Test Signal"],
        label_visibility="collapsed",
        key="audio_source_radio")
    
    x, Fs = None, None
    
    if audio_source == "Microphone":
        x, Fs = record_audio_from_mic()
        if x is not None:
            st.success(f"✓ Recorded: {len(x)} samples @ {Fs} Hz")
    
    elif audio_source == "Upload File":
        uploaded = st.file_uploader("Upload .wav file", type=["wav"])
        if uploaded:
            x, Fs = load_audio_from_file(uploaded)
            if x is not None:
                st.success(f"✓ Loaded: {len(x)} samples @ {Fs} Hz")
    
    else:
        test_signal_type = st.selectbox("Test Signal",
            ["Speech-like", "Pure Tone (440 Hz)", "Harmonic Complex", 
             "Noise", "Impulse", "Step Function"])
        
        if st.button("Generate Signal", width='stretch'):
            x = generate_test_signal(test_signal_type)
            Fs = 16000
            st.success(f"✓ Generated: {len(x)} samples @ {Fs} Hz")
    
    if x is not None and Fs is not None:
        x, Fs = preprocess_audio(x, Fs)
        
        st.divider()
        st.markdown("### 📊 SIGNAL INFO")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duration", f"{len(x)/Fs:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{Fs} Hz")

# ============================================================================
# TAB 1: ACOUSTIC ENGINE (MAIN SIMULATOR)
# ============================================================================

with nav_tabs[0]:
    st.markdown("## Acoustic Convolution Engine")
    st.markdown("Design impulse responses and simulate room acoustics in real-time")
    
    if x is None or Fs is None:
        st.warning("Please load or generate an input signal in the sidebar")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Echo Configuration")
        echo_mode = st.radio("Mode", 
            ["Custom Echo", "Preset Rooms"],
            label_visibility="collapsed",
            key="echo_mode_radio")
        
        if echo_mode == "Custom Echo":
            delay = st.slider("Delay (sec)", 0.1, 1.0, 0.3, 0.05)
            attenuation = st.slider("Attenuation", 0.1, 0.95, 0.5, 0.05)
            num_echoes = st.slider("Number of Echoes", 1, 5, 3)
            
            delay_samples = int(delay * Fs)
            h = np.zeros(delay_samples * (num_echoes + 1))
            h[0] = 1.0
            
            for echo_idx in range(1, num_echoes + 1):
                echo_pos = echo_idx * delay_samples
                if echo_pos < len(h):
                    h[echo_pos] = attenuation ** echo_idx
        else:
            room = st.selectbox("Room Type",
                ["Small Studio", "Medium Hall", "Large Concert Hall", 
                 "Cathedral", "Reverberant Chamber", "Anechoic Chamber"])
            
            h = generate_advanced_room_ir(room, Fs, duration=1.0)
            rt60 = compute_reverb_time(h, Fs)
            st.info(f"RT60 (Reverberation Time): **{rt60:.2f} seconds**")
    
    with col2:
        st.markdown("### Signal Processing Options")
        apply_filter_flag = st.checkbox("Apply Pre-Filter")
        
        if apply_filter_flag:
            filter_type = st.selectbox("Filter Type",
                ["Butterworth", "Chebyshev Type I", "Chebyshev Type II", "Elliptic"])
            
            filter_dict = {
                "Butterworth": "butterworth",
                "Chebyshev Type I": "cheby1",
                "Chebyshev Type II": "cheby2",
                "Elliptic": "ellip"
            }
            
            order = st.slider("Filter Order", 2, 8, 4)
            cutoff = st.slider("Cutoff Frequency (Hz)", 100, 8000, 2000)
            
            sos = design_digital_filter(filter_dict[filter_type], order, cutoff, Fs)
            x_processed = apply_filter(x, sos)
        else:
            x_processed = x.copy()
    
    # Perform convolution
    st.divider()
    
    if st.button("🔊 COMPUTE CONVOLUTION", key="conv_button", width='stretch'):
        y = signal.convolve(x_processed, h, mode='full')
        
        # Normalize output
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / (max_val * 1.2)
        
        # Time axes
        t_x = np.arange(len(x_processed)) / Fs
        t_h = np.arange(len(h)) / Fs
        t_y = np.arange(len(y)) / Fs
        
        # Create advanced visualization
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.patch.set_facecolor('white')
        
        # Plot 1: Input Signal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t_x, x_processed, linewidth=1.5, color='#667eea', alpha=0.85)
        ax1.fill_between(t_x, x_processed, alpha=0.2, color='#667eea')
        ax1.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        ax1.set_title('Input Signal x[n]', fontsize=12, fontweight='bold', loc='left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim([0, t_x[-1]])
        
        # Plot 2: Impulse Response
        ax2 = fig.add_subplot(gs[1, 0])
        markerline, stemlines, baseline = ax2.stem(t_h[:min(len(t_h), 500)], h[:min(len(h), 500)], basefmt=' ')
        stemlines.set_color('#2ca02c')
        stemlines.set_linewidth(1.5)
        markerline.set_color('#2ca02c')
        ax2.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        ax2.set_title('Impulse Response h[n]', fontsize=12, fontweight='bold', loc='left')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Output Signal
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(t_y, y, linewidth=1.2, color='#d62728', alpha=0.85)
        ax3.fill_between(t_y, y, alpha=0.2, color='#d62728')
        ax3.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
        ax3.set_title('Output Signal y[n] = x[n] * h[n]', fontsize=12, fontweight='bold', loc='left')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim([0, min(t_y[-1], 5.0)])
        
        # Plot 4: Frequency Response Comparison
        ax4 = fig.add_subplot(gs[2, :])
        
        freq_x, mag_x = fftfreq(len(x_processed), 1/Fs)[:len(x_processed)//2], np.abs(fft(x_processed))[:len(x_processed)//2]
        freq_y, mag_y = fftfreq(len(y), 1/Fs)[:len(y)//2], np.abs(fft(y))[:len(y)//2]
        
        ax4.semilogy(freq_x, 20*np.log10(mag_x + 1e-10), label='Input x[n]', linewidth=2, color='#667eea')
        ax4.semilogy(freq_y, 20*np.log10(mag_y + 1e-10), label='Output y[n]', linewidth=2, color='#d62728')
        ax4.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Magnitude (dB)', fontsize=10, fontweight='bold')
        ax4.set_title('Frequency Domain: Comb Filter Effect', fontsize=12, fontweight='bold', loc='left')
        ax4.grid(True, alpha=0.3, linestyle='--', which='both')
        ax4.legend(fontsize=10, loc='upper right')
        ax4.set_xlim([0, 8000])
        
        st.pyplot(fig, width='stretch')
        
        # Audio playback
        st.divider()
        st.markdown("### Audio Playback")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Original Signal**")
            st.audio(x_processed, sample_rate=Fs, format='audio/wav')
            st.caption(f"{len(x_processed)/Fs:.2f} sec")
        
        with col2:
            st.write("**Convolved Signal**")
            st.audio(y, sample_rate=Fs, format='audio/wav')
            st.caption(f"{len(y)/Fs:.2f} sec")
        
        with col3:
            st.write("**Impulse Response**")
            st.audio(h, sample_rate=Fs, format='audio/wav')
            st.caption(f"{len(h)/Fs:.2f} sec")
        
        # System metrics
        st.divider()
        st.markdown("### System Metrics")
        
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        
        with met_col1:
            st.markdown('<div class="metric-card"><div class="metric-label">IR Length</div><div class="metric-value">{:.3f} s</div></div>'.format(len(h)/Fs), unsafe_allow_html=True)
        
        with met_col2:
            st.markdown('<div class="metric-card"><div class="metric-label">Output Length</div><div class="metric-value">{:.3f} s</div></div>'.format(len(y)/Fs), unsafe_allow_html=True)
        
        with met_col3:
            max_output = np.max(np.abs(y))
            st.markdown('<div class="metric-card"><div class="metric-label">Peak Output</div><div class="metric-value">{:.3f}</div></div>'.format(max_output), unsafe_allow_html=True)
        
        with met_col4:
            snr_approx = 10 * np.log10(np.mean(y**2) + 1e-10)
            st.markdown('<div class="metric-card"><div class="metric-label">Power (dB)</div><div class="metric-value">{:.1f}</div></div>'.format(snr_approx), unsafe_allow_html=True)

# ============================================================================
# TAB 2: FILTER DESIGNER
# ============================================================================

with nav_tabs[1]:
    st.markdown("## Digital Filter Designer")
    st.markdown("Design and visualize IIR filters with real-time frequency response analysis")
    
    if x is None or Fs is None:
        st.warning("Load an input signal first")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filter_type = st.selectbox("Filter Family",
            ["Butterworth", "Chebyshev Type I", "Chebyshev Type II", "Elliptic"],
            key="filter_family_select")
    
    with col2:
        filter_subtype = st.selectbox("Filter Type",
            ["Lowpass", "Highpass", "Bandpass", "Bandstop"],
            key="filter_subtype_select")
    
    with col3:
        order = st.slider("Order", 2, 10, 5)
    
    with col4:
        ripple = st.slider("Ripple (dB)", 0.1, 3.0, 0.5) if filter_type != "Butterworth" else None
    
    # Cutoff frequency
    if filter_subtype in ["Lowpass", "Highpass"]:
        st.markdown("### Cutoff Frequency")
        cutoff = st.slider("Cutoff (Hz)", 100, Fs//2-100, 2000, 50)
        filter_dict = {"Butterworth": "butterworth", "Chebyshev Type I": "cheby1",
                       "Chebyshev Type II": "cheby2", "Elliptic": "ellip"}
        sos = design_digital_filter(filter_dict[filter_type], order, cutoff, Fs, filter_subtype.lower())
    else:
        st.markdown("### Frequency Range")
        col1, col2 = st.columns(2)
        with col1:
            f_low = st.slider("Lower Frequency (Hz)", 100, Fs//2-500, 500)
        with col2:
            f_high = st.slider("Upper Frequency (Hz)", f_low+100, Fs//2, 4000)
        
        filter_dict = {"Butterworth": "butterworth", "Chebyshev Type I": "cheby1",
                       "Chebyshev Type II": "cheby2", "Elliptic": "ellip"}
        sos = design_digital_filter(filter_dict[filter_type], order, (f_low, f_high), Fs, filter_subtype.lower())
    
    # Frequency response analysis
    st.divider()
    st.markdown("### Frequency Response Analysis")
    
    w, magnitude, phase, group_delay = compute_frequency_response(sos, Fs)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    # Magnitude response
    axes[0].plot(w, 20*np.log10(magnitude + 1e-10), linewidth=2.5, color='#667eea')
    axes[0].fill_between(w, 20*np.log10(magnitude + 1e-10), alpha=0.2, color='#667eea')
    axes[0].set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
    axes[0].set_title('Magnitude Response', fontsize=12, fontweight='bold', loc='left')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3dB line')
    axes[0].legend()
    axes[0].set_xlim([0, 8000])
    
    # Phase response
    axes[1].plot(w, np.unwrap(phase) * 180/np.pi, linewidth=2.5, color='#2ca02c')
    axes[1].set_ylabel('Phase (degrees)', fontsize=11, fontweight='bold')
    axes[1].set_title('Phase Response', fontsize=12, fontweight='bold', loc='left')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim([0, 8000])
    
    # Group delay
    axes[2].plot(w[:-1], group_delay, linewidth=2.5, color='#d62728')
    axes[2].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Group Delay (samples)', fontsize=11, fontweight='bold')
    axes[2].set_title('Group Delay', fontsize=12, fontweight='bold', loc='left')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xlim([0, 8000])
    
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    
    # Test filter on signal
    st.divider()
    st.markdown("### Apply Filter to Signal")
    
    if st.button("Apply Filter", width='stretch'):
        filtered = apply_filter(x, sos)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Signal**")
            st.audio(x, sample_rate=Fs, format='audio/wav')
        
        with col2:
            st.write("**Filtered Signal**")
            st.audio(filtered, sample_rate=Fs, format='audio/wav')

# ============================================================================
# TAB 3: ADVANCED ANALYSIS
# ============================================================================

with nav_tabs[2]:
    st.markdown("## Advanced Signal Analysis")
    st.markdown("Detailed spectral and temporal analysis with multiple perspectives")
    
    if x is None or Fs is None:
        st.warning("Load an input signal first")
        st.stop()
    
    analysis_type = st.selectbox("Analysis Type",
        ["Spectrogram", "Power Spectral Density", "Autocorrelation", 
         "Zero Crossings", "Cepstrum"],
        key="analysis_type_select")
    
    st.divider()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    if analysis_type == "Spectrogram":
        f, t, Sxx = signal.spectrogram(x, Fs, nperseg=1024)
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_title('Spectrogram', fontsize=13, fontweight='bold', loc='left')
        ax.set_ylim([0, 8000])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontweight='bold')
    
    elif analysis_type == "Power Spectral Density":
        freqs, psd = signal.welch(x, Fs, nperseg=1024)
        ax.semilogy(freqs, psd, linewidth=2, color='#667eea')
        ax.fill_between(freqs, psd, alpha=0.2, color='#667eea')
        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Power/Frequency (V^2/Hz)', fontweight='bold')
        ax.set_title('Power Spectral Density (Welch Method)', fontsize=13, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([0, 8000])
    
    elif analysis_type == "Autocorrelation":
        acf = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        lags = np.arange(len(acf)) / Fs
        ax.plot(lags[:int(0.5*Fs)], acf[:int(0.5*Fs)], linewidth=2, color='#2ca02c')
        ax.fill_between(lags[:int(0.5*Fs)], acf[:int(0.5*Fs)], alpha=0.2, color='#2ca02c')
        ax.set_xlabel('Lag (s)', fontweight='bold')
        ax.set_ylabel('Autocorrelation', fontweight='bold')
        ax.set_title('Autocorrelation Function', fontsize=13, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
    
    elif analysis_type == "Zero Crossings":
        zero_crossings = np.where(np.diff(np.sign(x)))[0]
        ax.plot(x, linewidth=1.5, color='#667eea', label='Signal')
        ax.scatter(zero_crossings, x[zero_crossings], color='#d62728', s=50, zorder=5, label='Zero Crossings')
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.set_title(f'Zero Crossings Detection ({len(zero_crossings)} crossings)', fontsize=13, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, min(5000, len(x))])
    
    elif analysis_type == "Cepstrum":
        X = fft(x)
        cepstrum = np.real(ifft(np.log(np.abs(X) + 1e-10)))
        ax.plot(cepstrum[:2000], linewidth=2, color='#d62728')
        ax.fill_between(range(2000), cepstrum[:2000], alpha=0.2, color='#d62728')
        ax.set_xlabel('Quefrency (samples)', fontweight='bold')
        ax.set_ylabel('Cepstral Magnitude', fontweight='bold')
        ax.set_title('Cepstrum', fontsize=13, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig, width='stretch')

# ============================================================================
# TAB 4: ROOM ACOUSTICS
# ============================================================================

with nav_tabs[3]:
    st.markdown("## Advanced Room Acoustics Simulator")
    st.markdown("Explore acoustical properties of various environments")
    
    if x is None or Fs is None:
        st.warning("Load an input signal first")
        st.stop()
    
    room_type = st.selectbox("Select Room",
        ["Anechoic Chamber", "Small Studio", "Medium Hall", 
         "Large Concert Hall", "Cathedral", "Reverberant Chamber"],
        key="room_select")
    
    # Generate IR and compute metrics
    h = generate_advanced_room_ir(room_type, Fs, duration=2.0)
    rt60 = compute_reverb_time(h, Fs)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Room Type</div><div class="metric-value" style="font-size:1.3rem">{room_type}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">RT60</div><div class="metric-value">{rt60:.2f}s</div></div>', unsafe_allow_html=True)
    with col3:
        ir_energy = np.sum(h**2)
        st.markdown(f'<div class="metric-card"><div class="metric-label">IR Energy</div><div class="metric-value">{ir_energy:.3f}</div></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # IR visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    
    t_h = np.arange(len(h)) / Fs
    
    # Time domain
    axes[0, 0].plot(t_h, h, linewidth=1.5, color='#667eea')
    axes[0, 0].fill_between(t_h, h, alpha=0.2, color='#667eea')
    axes[0, 0].set_xlabel('Time (s)', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude', fontweight='bold')
    axes[0, 0].set_title('Impulse Response (Time Domain)', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy decay curve
    envelope = np.abs(h)
    envelope = np.maximum(envelope, 1e-10)
    energy_db = 10 * np.log10(envelope ** 2)
    axes[0, 1].plot(t_h, energy_db, linewidth=2, color='#2ca02c')
    axes[0, 1].axhline(np.max(energy_db) - 60, color='r', linestyle='--', label='RT60 threshold')
    axes[0, 1].set_xlabel('Time (s)', fontweight='bold')
    axes[0, 1].set_ylabel('Energy (dB)', fontweight='bold')
    axes[0, 1].set_title('Energy Decay Curve', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Frequency response
    freq_h, mag_h = fftfreq(len(h), 1/Fs)[:len(h)//2], np.abs(fft(h))[:len(h)//2]
    axes[1, 0].semilogy(freq_h, mag_h, linewidth=2, color='#d62728')
    axes[1, 0].set_xlabel('Frequency (Hz)', fontweight='bold')
    axes[1, 0].set_ylabel('Magnitude', fontweight='bold')
    axes[1, 0].set_title('Magnitude Spectrum', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, which='both')
    axes[1, 0].set_xlim([0, 8000])
    
    # Cumulative energy
    cumsum = np.cumsum(h**2) / np.sum(h**2) * 100
    axes[1, 1].plot(t_h, cumsum, linewidth=2, color='#764ba2')
    axes[1, 1].axhline(95, color='r', linestyle='--', label='95% energy')
    axes[1, 1].set_xlabel('Time (s)', fontweight='bold')
    axes[1, 1].set_ylabel('Cumulative Energy (%)', fontweight='bold')
    axes[1, 1].set_title('Cumulative Energy', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    
    # Apply to signal
    st.divider()
    if st.button("Simulate Room Acoustics", width='stretch'):
        y = signal.convolve(x, h, mode='full')
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / (max_val * 1.2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original**")
            st.audio(x, sample_rate=Fs)
        with col2:
            st.write(f"**In {room_type}**")
            st.audio(y, sample_rate=Fs)

# ============================================================================
# TAB 5: EFFECTS LAB
# ============================================================================

with nav_tabs[4]:
    st.markdown("## Audio Effects Laboratory")
    st.markdown("Apply real-time audio effects and transformations")
    
    if x is None or Fs is None:
        st.warning("Load an input signal first")
        st.stop()
    
    effect = st.selectbox("Effect",
        ["Vibrato", "Tremolo", "Distortion", "Chorus", "Echo/Delay", "Reverb"],
        key="effect_select")
    
    st.divider()
    
    if effect == "Vibrato":
        rate = st.slider("Rate (Hz)", 2, 15, 5)
        depth = st.slider("Depth", 0.01, 0.5, 0.1)
        processed = generate_special_effects(x, Fs, 'vibrato', rate=rate, depth=depth)
    
    elif effect == "Tremolo":
        rate = st.slider("Rate (Hz)", 2, 15, 5)
        depth = st.slider("Depth", 0.1, 1.0, 0.5)
        processed = generate_special_effects(x, Fs, 'tremolo', rate=rate, depth=depth)
    
    elif effect == "Distortion":
        gain = st.slider("Gain", 1, 20, 5)
        processed = generate_special_effects(x, Fs, 'distortion', gain=gain)
    
    elif effect == "Chorus":
        delay_ms = st.slider("Delay (ms)", 10, 100, 50)
        depth = st.slider("Depth", 0.0, 1.0, 0.3)
        processed = generate_special_effects(x, Fs, 'chorus', delay_ms=delay_ms, depth=depth)
    
    elif effect == "Echo/Delay":
        delay = st.slider("Delay (seconds)", 0.1, 1.0, 0.3)
        feedback = st.slider("Feedback", 0.0, 0.95, 0.5)
        
        delay_samples = int(delay * Fs)
        processed = np.zeros(len(x) + delay_samples * 3)
        processed[:len(x)] = x
        
        for i in range(3):
            delayed_pos = (i + 1) * delay_samples
            processed[delayed_pos:delayed_pos + len(x)] += (feedback ** (i + 1)) * x
        
        processed = processed[:len(x) + delay_samples * 3]
    
    else:  # Reverb
        room_type = st.selectbox("Reverb Type",
            ["Small Room", "Medium Hall", "Large Hall"],
            key="reverb_type_select")
        
        h = generate_advanced_room_ir(room_type, Fs, duration=1.0)
        processed = signal.convolve(x, h, mode='full')[:len(x) + len(h)]
        max_val = np.max(np.abs(processed))
        if max_val > 0:
            processed = processed / (max_val * 1.2)
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    t = np.arange(len(x)) / Fs
    t_proc = np.arange(len(processed)) / Fs
    
    axes[0].plot(t, x, linewidth=1.5, color='#667eea', label='Original')
    axes[0].fill_between(t, x, alpha=0.2, color='#667eea')
    axes[0].set_ylabel('Amplitude', fontweight='bold')
    axes[0].set_title('Original Signal', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, min(t[-1], 2)])
    
    axes[1].plot(t_proc, processed, linewidth=1.5, color='#d62728', label='Processed')
    axes[1].fill_between(t_proc, processed, alpha=0.2, color='#d62728')
    axes[1].set_xlabel('Time (s)', fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontweight='bold')
    axes[1].set_title(f'{effect} Effect Applied', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, min(t_proc[-1], 2)])
    
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    
    # Audio playback
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original**")
        st.audio(x, sample_rate=Fs)
    
    with col2:
        st.write(f"**With {effect}**")
        st.audio(processed, sample_rate=Fs)

# ============================================================================
# TAB 6: THEORY & LEARNING
# ============================================================================

with nav_tabs[5]:
    st.markdown("## 📚 Signals & Systems Theory")
    st.markdown("Complete mathematical foundation for DSP concepts")
    
    theory_section = st.selectbox("Select Topic",
        ["Fundamentals", "Convolution", "Fourier Analysis", 
         "Filter Design", "Room Acoustics", "Spectral Analysis"],
        key="theory_section_select")
    
    st.divider()
    
    if theory_section == "Fundamentals":
        st.markdown(r"""
        ### Digital Signals: From Continuous to Discrete
        
        Audio is fundamentally a **continuous-time signal**: variations in air pressure over time.
        To process it digitally, we must **sample** the continuous signal at discrete time intervals.
        
        #### The Sampling Process
        
        At a fixed sampling rate $F_s$, we measure signal amplitude at intervals $T = 1/F_s$:
        
        $$x[n] = x_c(nT) \quad \text{where} \quad n = 0, 1, 2, \ldots$$
        
        #### Nyquist-Shannon Sampling Theorem
        
        To faithfully represent frequencies up to frequency $f_{max}$, the sampling rate must satisfy:
        
        $$F_s \geq 2 f_{max}$$
        
        The maximum representable frequency is the **Nyquist frequency**: $f_{Nyquist} = F_s/2$
        
        #### Why This Matters
        
        - **CD quality**: $F_s = 44,100$ Hz → can capture frequencies up to **22 kHz** (beyond human hearing ~20 kHz)
        - **Telephony**: $F_s = 8,000$ Hz → captures up to **4 kHz** (sufficient for speech)
        - **Professional audio**: $F_s = 96,000$ Hz or higher for high-quality production
        """)
    
    elif theory_section == "Convolution":
        st.markdown(r"""
        ### Convolution: The Heart of DSP
        
        Convolution is the mathematical operation that describes how a Linear, Time-Invariant (LSI) system 
        responds to any input signal.
        
        #### Definition
        
        For discrete-time signals, the convolution sum is:
        
        $$y[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k] = x[n] \circledast h[n]$$
        
        where:
        - $x[n]$ is the input signal
        - $h[n]$ is the system's impulse response
        - $y[n]$ is the output signal
        
        #### Conceptual Interpretation
        
        1. **Folding**: Reverse $h[k]$ to create $h[-k]$
        2. **Shifting**: Slide the folded impulse response to index $n$
        3. **Multiplication**: Multiply element-wise with input $x[k]$
        4. **Summation**: Add all products to get $y[n]$
        
        This operation uniquely characterizes the entire system response!
        
        #### Computational Complexity
        
        - Direct convolution: **O(N²)** operations
        - FFT-based convolution: **O(N log N)** operations (vastly faster for long signals!)
        """)
    
    elif theory_section == "Fourier Analysis":
        st.markdown(r"""
        ### The Fourier Transform: Time ↔ Frequency
        
        The Discrete Fourier Transform (DFT) reveals the frequency content of a signal.
        
        #### DFT Definition
        
        $$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}$$
        
        Result: A complex number $X[k]$ for each frequency bin $k$
        - **Magnitude** $|X[k]|$: How much energy at frequency $k$
        - **Phase** $\angle X[k]$: The phase relationship
        
        #### The Convolution-Multiplication Duality
        
        A remarkable property bridges time and frequency domains:
        
        $$y[n] = x[n] \circledast h[n] \quad \Longleftrightarrow \quad Y[k] = X[k] \cdot H[k]$$
        
        **This is why we can apply filtering in either domain!**
        
        #### Practical Implication
        
        Instead of computing the convolution sum (slow!), we can:
        1. Transform to frequency domain using FFT (fast!)
        2. Multiply spectra pointwise (very fast!)
        3. Inverse transform back (fast!)
        
        **Total cost**: $O(N \\log N)$ instead of $O(N^2)$
        """)
    
    elif theory_section == "Filter Design":
        st.markdown(r"""
        ### Digital Filter Design
        
        Filters selectively remove, emphasize, or shape frequencies in a signal.
        
        #### IIR vs FIR Filters
        
        **Infinite Impulse Response (IIR)**:
        - Feedback from output to input
        - Compact designs (fewer coefficients)
        - Phase distortion possible
        - Examples: Butterworth, Chebyshev, Elliptic
        
        **Finite Impulse Response (FIR)**:
        - No feedback (feedforward only)
        - Linear phase possible
        - Requires more coefficients
        - Always stable
        
        #### Common Filter Types
        
        | Filter | Response | Characteristic |
        |--------|----------|-----------------|
        | **Butterworth** | Maximally flat | Best for audio (no ripple) |
        | **Chebyshev I** | Ripple in passband | Steeper rolloff |
        | **Chebyshev II** | Ripple in stopband | Better stopband attenuation |
        | **Elliptic** | Ripple both bands | Sharpest transition |
        
        #### Key Parameters
        
        - **Order**: Higher = steeper rolloff but more phase distortion
        - **Cutoff Frequency**: Where -3dB point occurs
        - **Ripple**: Allowed magnitude variation in passband/stopband
        """)
    
    elif theory_section == "Room Acoustics":
        st.markdown(r"""
        ### Room Acoustics: Impulse Response of Spaces
        
        Every room has a unique acoustic signature captured by its **impulse response** $h[n]$.
        
        #### Structure of Room IR
        
        A typical room impulse response consists of:
        
        1. **Direct Sound**: The primary impulse at $h[0]$
        2. **Early Reflections**: Discrete echoes from walls, floor, ceiling
        3. **Late Reverb**: Dense, diffuse reflections creating smooth decay
        
        #### The Reverberation Time (RT60)
        
        The time for sound energy to decay by 60 dB:
        
        $$\\text{RT60} = \\text{Time for energy to drop 60 dB below peak}$$
        
        **Interpretation**:
        - Small room: 0.2-0.5 s (dry, absorptive)
        - Concert hall: 1.5-2.5 s (lively, reverberant)
        - Cathedral: 3-5+ s (very reverberant)
        
        #### The Comb Filter Effect
        
        When echoes are periodic (delay = $D$ samples), the frequency response exhibits 
        peaks and nulls spaced by $\\Delta f = F_s / D$:
        
        $$h[n] = \\delta[n] + \\alpha \\delta[n-D] + \\alpha^2 \\delta[n-2D] + \\cdots$$
        
        This creates "coloration" - certain frequencies are boosted while others are attenuated!
        """)
    
    else:  # Spectral Analysis
        st.markdown(r"""
        ### Spectral Analysis Techniques
        
        Modern audio processing relies on multiple frequency-domain representations.
        
        #### Power Spectral Density (PSD)
        
        Shows the distribution of power across frequencies:
        
        $$S_{xx}(f) = |X(f)|^2$$
        
        **Welch's Method**: Averages multiple overlapping segments for noise reduction.
        
        #### Spectrogram
        
        Time-frequency representation showing how spectrum evolves:
        
        $$S(t, f) = |X_t(f)|^2$$
        
        where $X_t(f)$ is the FFT of a windowed segment at time $t$.
        
        #### Autocorrelation
        
        Measures similarity of signal with delayed versions:
        
        $$R(\\tau) = \\sum_n x[n] x[n+\\tau]$$
        
        **Applications**:
        - Pitch detection (fundamental frequency)
        - Echo detection
        - Periodicity analysis
        
        #### Cepstrum
        
        The "spectrum of the spectrum" - reveals harmonic structure:
        
        $$c[n] = \\text{IFFT}(\\log |\\text{FFT}(x[n])|)$$
        
        **Application**: Separating voice formants from pitch in speech processing.
        """)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 3rem;'>
    <p><strong>LSI Acoustic Studio PRO</strong> — Professional Educational Platform</p>
    <p>Advanced DSP, Room Acoustics, and Real-time Audio Processing</p>
</div>
""", unsafe_allow_html=True)
# coding: utf-8
"""
LSI Acoustic Studio - Interactive DSP Learning Platform
A university-level educational tool for Signals and Systems concepts
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LSI Acoustic Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #2ca02c;
        margin-top: 1.5rem;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #d62728;
    }
    .metric-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_audio_from_mic():
    """Record audio from microphone using st.audio_input."""
    audio_data = st.audio_input("Record your voice")
    if audio_data is not None:
        try:
            audio_bytes = audio_data.getvalue()
            data, sr = sf.read(io.BytesIO(audio_bytes))
            return data, sr
        except Exception as e:
            st.sidebar.error(f"Error reading microphone input: {e}")
            return None, None
    return None, None


def load_audio_from_file(uploaded_file):
    """Load audio from uploaded .wav file."""
    if uploaded_file is not None:
        try:
            data, sr = sf.read(uploaded_file)
            return data, sr
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            return None, None
    return None, None


def generate_default_audio(duration: float = 2.0, sr: int = 16000):
    """Generate a synthetic test signal combining a chirp and noise burst."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Frequency sweep from 200 Hz to 2000 Hz
    f_start, f_end = 200, 2000
    chirp_signal = signal.chirp(t, f_start, duration, f_end, method='linear')
    
    # Add a noise burst in the middle (0.5 to 1.0 seconds)
    noise_burst = np.zeros_like(t)
    noise_start = int(0.5 * sr)
    noise_end = int(1.0 * sr)
    noise_burst[noise_start:noise_end] = np.random.normal(0, 0.3, noise_end - noise_start)
    
    # Combine signals
    combined_signal = 0.7 * chirp_signal + 0.3 * noise_burst
    
    return combined_signal, sr


def preprocess_audio(audio_data: np.ndarray, sr: int):
    """Normalize audio to [-1, 1] and convert stereo to mono."""
    # Convert stereo to mono by averaging channels
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)
    
    # Ensure 1D array
    audio_data = np.atleast_1d(audio_data.flatten())
    
    # Normalize to [-1, 1]
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    return audio_data, sr


def generate_echo_impulse_response(delay: float, attenuation: float, sr: int):
    """
    Generate impulse response for echo effect (Math Sandbox mode).
    h[n] = delta[n] + alpha * delta[n-D] + alpha^2 * delta[n-2D] + ...
    """
    delay_samples = int(delay * sr)
    max_samples = delay_samples * 4
    h = np.zeros(max_samples)
    
    h[0] = 1.0  # Primary impulse
    
    # Add 3 echoes with exponential decay
    for echo_idx in range(1, 4):
        echo_pos = echo_idx * delay_samples
        if echo_pos < len(h):
            h[echo_pos] = attenuation ** echo_idx
    
    return h


def generate_room_impulse_response(room_type: str, sr: int, duration: float = 0.5):
    """
    Simulate room impulse response for Real-World Vault mode.
    Generates different characteristics for Small Room, Large Hall, and Deep Cave.
    """
    num_samples = int(duration * sr)
    h = np.zeros(num_samples)
    
    if room_type == "Small Room":
        # Short, quick decay with minimal reflections
        decay_envelope = np.exp(-np.linspace(0, 5, num_samples))
        early_reflections = np.random.normal(0, 0.1, num_samples)
        h = decay_envelope * (0.8 + 0.2 * early_reflections)
        h[0] = 1.0
        
    elif room_type == "Large Hall":
        # Longer decay with structured early reflections
        decay_envelope = np.exp(-np.linspace(0, 8, num_samples))
        reflections = np.zeros(num_samples)
        # Add discrete early reflections at specific delays
        reflection_indices = [
            int(0.02 * sr), int(0.05 * sr), 
            int(0.08 * sr), int(0.12 * sr)
        ]
        for idx in reflection_indices:
            if idx < num_samples:
                reflections[idx] = 0.5 + np.random.normal(0, 0.1)
        h = decay_envelope * (0.6 + 0.4 * reflections)
        h[0] = 1.0
        
    elif room_type == "Deep Cave":
        # Highly diffuse response with very long tail
        decay_envelope = np.exp(-np.linspace(0, 10, num_samples))
        diffuse_reflections = np.random.normal(0.2, 0.15, num_samples)
        h = decay_envelope * np.abs(diffuse_reflections)
        h[0] = 1.0
    
    # Normalize
    h = h / (np.max(np.abs(h)) + 1e-10)
    return h


def convolve_signals(x: np.ndarray, h: np.ndarray):
    """Perform linear convolution using scipy.signal.convolve."""
    return signal.convolve(x, h, mode='full')


def compute_magnitude_spectrum(signal_data: np.ndarray, sr: int):
    """
    Compute magnitude spectrum using FFT.
    Returns frequencies and magnitudes up to 8000 Hz.
    """
    # Compute FFT
    fft_result = fft(signal_data)
    magnitude = np.abs(fft_result)
    
    # Compute frequency axis
    freqs = fftfreq(len(signal_data), 1/sr)
    
    # Keep only positive frequencies up to 8000 Hz
    positive_freq_mask = (freqs >= 0) & (freqs <= 8000)
    
    return freqs[positive_freq_mask], magnitude[positive_freq_mask]


# ============================================================================
# SIDEBAR: AUDIO INPUT & MODE SELECTION
# ============================================================================

st.sidebar.markdown("# Configuration Panel")

st.sidebar.markdown("## Audio Input")

# Audio input method selection
audio_input_method = st.sidebar.radio(
    "Choose input signal x[n]:",
    ["Microphone Recording", "Upload File", "Default Test Signal"],
    help="Select how to provide the input audio signal"
)

x = None
Fs = None
audio_loaded = False

if audio_input_method == "Microphone Recording":
    x, Fs = record_audio_from_mic()
    if x is not None:
        audio_loaded = True
    else:
        st.sidebar.info("No recording captured yet. Please record audio.")

elif audio_input_method == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a .wav file", 
        type=["wav"],
        help="Select a WAV file from your computer"
    )
    if uploaded_file:
        x, Fs = load_audio_from_file(uploaded_file)
        if x is not None:
            audio_loaded = True

else:  # Default Test Signal
    if st.sidebar.button("Generate Default Test Signal", width='stretch'):
        x, Fs = generate_default_audio()
        audio_loaded = True
        st.sidebar.success("Test signal generated!")

# Preprocess audio if available
if audio_loaded and x is not None and Fs is not None:
    x, Fs = preprocess_audio(x, Fs)
    st.sidebar.success(f"Audio loaded: {len(x)} samples @ {Fs} Hz")

st.sidebar.divider()

# Application mode selection
st.sidebar.markdown("## LSI System Mode")

mode = st.sidebar.radio(
    "Select System Mode:",
    ["Mode A: Math Sandbox", "Mode B: Real-World Vault"],
    help="Choose how to generate the impulse response"
)

h = None

if mode == "Mode A: Math Sandbox":
    st.sidebar.markdown("### Echo Parameters")
    
    delay = st.sidebar.slider(
        "Echo Delay (D) in seconds",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Time delay between echoes"
    )
    
    attenuation = st.sidebar.slider(
        "Attenuation (Alpha) alpha",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Decay factor per echo (0 < alpha < 1)"
    )
    
    if audio_loaded and x is not None and Fs is not None:
        h = generate_echo_impulse_response(delay, attenuation, Fs)
        st.sidebar.info(f"IR generated: {len(h)} samples ({len(h)/Fs:.3f} sec)")

else:  # Mode B: Real-World Vault
    st.sidebar.markdown("### Room Configuration")
    
    room_type = st.sidebar.selectbox(
        "Select Room Type:",
        ["Small Room", "Large Hall", "Deep Cave"],
        help="Choose an acoustic environment"
    )
    
    if audio_loaded and x is not None and Fs is not None:
        h = generate_room_impulse_response(room_type, Fs)
        st.sidebar.info(f"IR generated: {len(h)} samples ({len(h)/Fs:.3f} sec)")

st.sidebar.divider()

# Convolve button
process_audio = st.sidebar.button(
    "Convolve!", 
    width='stretch',
    type="primary",
    help="Click to compute convolution and update visualizations"
)

# ============================================================================
# MAIN CONTENT: TABS
# ============================================================================

st.title("LSI Acoustic Studio")
st.markdown("**An Interactive DSP Learning Platform for Signals and Systems**")
st.divider()

# Check if audio and impulse response are available
if x is None or Fs is None:
    st.warning("No audio signal loaded. Please provide an input signal using the sidebar!")
    st.stop()

if h is None:
    st.warning("No impulse response generated. Please select a system mode in the sidebar!")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["Acoustic Simulator", "Signals & Systems Theory"])

# ============================================================================
# TAB 1: ACOUSTIC SIMULATOR
# ============================================================================

with tab1:
    st.header("Acoustic Simulator - Time and Frequency Domain Analysis")
    
    if process_audio:
        # Perform linear convolution
        y = convolve_signals(x, h)
        
        # Time axes
        t_x = np.arange(len(x)) / Fs
        t_h = np.arange(len(h)) / Fs
        t_y = np.arange(len(y)) / Fs
        
        # Create two-column layout for time and frequency domains
        col1, col2 = st.columns(2)
        
        # --- TIME DOMAIN ANALYSIS ---
        with col1:
            st.subheader("Time Domain Analysis")
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 11))
            fig.patch.set_facecolor('white')
            
            # Plot 1: Input Signal
            axes[0].plot(t_x, x, linewidth=1.2, color='#1f77b4', alpha=0.9)
            axes[0].fill_between(t_x, x, alpha=0.2, color='#1f77b4')
            axes[0].set_xlabel('Time (seconds)', fontsize=10)
            axes[0].set_ylabel('Amplitude', fontsize=10)
            axes[0].set_title('Input Signal x[n]', fontsize=11, fontweight='bold')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].set_xlim([0, t_x[-1]])
            axes[0].axhline(y=0, color='k', linewidth=0.5)
            
            # Plot 2: Impulse Response (stem plot)
            stem_container = axes[1].stem(t_h, h, basefmt=' ')
            stem_container.markerline.set_color('#2ca02c')
            stem_container.markerline.set_markersize(8)
            stem_container.stemlines.set_color('#2ca02c')
            stem_container.stemlines.set_linewidth(1.5)
            axes[1].set_xlabel('Time (seconds)', fontsize=10)
            axes[1].set_ylabel('Amplitude', fontsize=10)
            axes[1].set_title('Impulse Response h[n]', fontsize=11, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].set_xlim([0, min(t_h[-1], 1.0)])
            axes[1].axhline(y=0, color='k', linewidth=0.5)
            
            # Plot 3: Output Signal
            axes[2].plot(t_y, y, linewidth=1.2, color='#d62728', alpha=0.9)
            axes[2].fill_between(t_y, y, alpha=0.2, color='#d62728')
            axes[2].set_xlabel('Time (seconds)', fontsize=10)
            axes[2].set_ylabel('Amplitude', fontsize=10)
            axes[2].set_title('Output Signal y[n] = x[n] * h[n]', fontsize=11, fontweight='bold')
            axes[2].grid(True, alpha=0.3, linestyle='--')
            axes[2].set_xlim([0, t_y[-1]])
            axes[2].axhline(y=0, color='k', linewidth=0.5)
            
            plt.tight_layout()
            st.pyplot(fig, width='stretch')
        
        # --- FREQUENCY DOMAIN ANALYSIS ---
        with col2:
            st.subheader("Frequency Domain Analysis")
            
            # Compute magnitude spectra
            freq_x, mag_x = compute_magnitude_spectrum(x, Fs)
            freq_y, mag_y = compute_magnitude_spectrum(y, Fs)
            
            # Convert to dB scale
            mag_x_db = 20 * np.log10(mag_x + 1e-10)
            mag_y_db = 20 * np.log10(mag_y + 1e-10)
            
            # Plot overlaid spectra
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')
            
            ax.plot(freq_x, mag_x_db, label='Input x[n]', 
                   linewidth=1.8, color='#1f77b4', alpha=0.8)
            ax.plot(freq_y, mag_y_db, label='Output y[n]', 
                   linewidth=1.8, color='#d62728', alpha=0.8)
            
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('Magnitude (dB)', fontsize=10)
            ax.set_title('Magnitude Spectrum: Comb Filter Effect', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='upper right')
            ax.set_xlim([0, 8000])
            
            plt.tight_layout()
            st.pyplot(fig, width='stretch')
        
        # --- AUDIO PLAYBACK ---
        st.divider()
        st.subheader("Audio Playback")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Signal**")
            st.audio(x, sample_rate=Fs, format='audio/wav')
            st.caption(f"Duration: {len(x)/Fs:.2f} sec")
        
        with col2:
            st.write("**Convolved Signal**")
            y_normalized = y / (np.max(np.abs(y)) + 1e-10)
            st.audio(y_normalized, sample_rate=Fs, format='audio/wav')
            st.caption(f"Duration: {len(y)/Fs:.2f} sec")
        
        # --- SYSTEM INFORMATION ---
        st.divider()
        st.subheader("System Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sampling Rate", f"{Fs:,} Hz")
        with col2:
            st.metric("Input Length", f"{len(x):,} samples")
        with col3:
            st.metric("IR Length", f"{len(h):,} samples")
        with col4:
            st.metric("Output Length", f"{len(y):,} samples")
        
        # Display convolution formula
        st.divider()
        st.markdown("**Convolution Computation:**")
        st.latex(r"y[n] = \sum_{k=0}^{M-1} x[k] \cdot h[n-k]")
        st.markdown(f"where input length = {len(x)}, IR length = {len(h)}, "
                   f"output length = {len(y)}")
    
    else:
        st.info("Configure the system in the sidebar and click 'Convolve!' to start the analysis.")


# ============================================================================
# TAB 2: SIGNALS & SYSTEMS THEORY
# ============================================================================

with tab2:
    st.header("Signals & Systems Theory")
    st.markdown("""
    This section provides the mathematical foundation for understanding the acoustic 
    simulator and the DSP concepts it demonstrates.
    """)
    
    # --- SECTION 1 ---
    st.markdown("## 1. Audio as a Discrete-Time Signal")
    st.markdown(r"""
    Continuous audio - sound pressure waves in air - is converted into a discrete-time signal 
    through a process called **sampling**. At a fixed sampling rate F_s (typically 16,000 Hz 
    for telephony, 44,100 Hz for CD-quality audio), we measure the air pressure amplitude 
    at regular time intervals T = 1/F_s:
    """)
    
    st.latex(r"x[n] = x_c(nT) \quad \text{where} \quad n = 0, 1, 2, \ldots \quad \text{and} \quad T = \frac{1}{F_s}")
    
    st.markdown(r"""
    The resulting discrete sequence x[n] is a sequence of real numbers, each representing 
    the amplitude of the audio signal at a specific point in time.
    
    **The Nyquist-Shannon Sampling Theorem** guarantees that if we sample a continuous signal 
    at rate F_s, we can faithfully reconstruct frequencies up to the **Nyquist frequency**:
    """)
    
    st.latex(r"f_{\text{Nyquist}} = \frac{F_s}{2}")
    
    st.markdown(r"""
    Any frequency component higher than f_Nyquist will be aliased (incorrectly 
    represented) in the discrete signal. This is why microphone inputs are filtered before 
    sampling to remove high-frequency components.
    """)
    
    st.divider()
    
    # --- SECTION 2 ---
    st.markdown("## 2. The Room as a Linear Shift-Invariant (LSI) System")
    st.markdown(r"""
    A physical room behaves as a **Linear Shift-Invariant (LSI) system**. This means:
    
    **Linearity:** The system respects superposition. If input x_1[n] produces output y_1[n] 
    and input x_2[n] produces output y_2[n], then:
    """)
    
    st.latex(r"ax_1[n] + bx_2[n] \quad \rightarrow \quad ay_1[n] + by_2[n]")
    
    st.markdown(r"""
    **Time-Invariance (Shift-Invariance):** The system's behavior does not change over time. 
    If input x[n] produces output y[n], then a delayed input x[n - n_0] produces a 
    proportionally delayed output:
    """)
    
    st.latex(r"x[n - n_0] \quad \rightarrow \quad y[n - n_0]")
    
    st.markdown(r"""
    The complete behavior of an LSI system is fully captured by a single function: 
    its **Impulse Response** h[n]. This is the output of the system when the input is 
    the **Dirac delta function** delta[n]:
    """)
    
    st.latex(r"\delta[n] = \begin{cases} 1 & \text{if } n = 0 \\ 0 & \text{if } n \neq 0 \end{cases}")
    
    st.markdown(r"""
    In a room, the impulse response captures the primary sound plus all echoes from reflections:
    """)
    
    st.latex(r"h[n] = \delta[n] + \alpha \delta[n - D] + \alpha^2 \delta[n - 2D] + \alpha^3 \delta[n - 3D] + \cdots")
    
    st.markdown(r"""
    where:
    - D is the delay (in samples) corresponding to the distance a sound wave travels before the first echo
    - alpha (with 0 < alpha < 1) is the **attenuation factor**, representing how much energy is lost per reflection
    
    For example, if the room is 34 meters long and sound travels at 340 m/s with F_s = 16,000 Hz, 
    then D = 34 / 340 * 16,000 = 1,600 samples.
    """)
    
    st.divider()
    
    # --- SECTION 3 ---
    st.markdown("## 3. The Convolution Sum")
    st.markdown(r"""
    Given an input signal x[n] and the impulse response h[n] of an LSI system, 
    the output signal y[n] is computed via the **discrete-time convolution sum**:
    """)
    
    st.latex(r"y[n] = \sum_{k=-\infty}^{\infty} x[k] \, h[n - k] = \sum_{k=-\infty}^{\infty} x[n-k] \, h[k]")
    
    st.markdown(r"""
    Conceptually, this operation consists of four steps:
    
    1. **Folding:** Reverse the impulse response in time to create h[-k].
    2. **Shifting:** Shift the folded impulse response by n positions to align with the current time.
    3. **Multiplication:** Multiply the shifted impulse response element-wise with the input signal.
    4. **Summation:** Sum all products to produce the single output sample y[n].
    
    This process is repeated for every output index n, producing the complete output sequence y[n].
    
    ### Example: Echo with Attenuation
    
    For a simple echo system where h[n] = delta[n] + 0.5 * delta[n - D], 
    the convolution produces:
    """)
    
    st.latex(r"y[n] = x[n] + 0.5 \, x[n - D]")
    
    st.markdown(r"""
    This is exactly what we expect: the original signal plus an attenuated copy delayed by D samples.
    
    **Importance:** Convolution is the fundamental operation in signal processing. 
    Once you know h[n], you can compute the system's output for *any* input signal via convolution. 
    No matter how complex the input, the convolution operation fully describes the system's response.
    """)
    
    st.divider()
    
    # --- SECTION 4 ---
    st.markdown("## 4. Frequency Domain Characterization")
    st.markdown(r"""
    While time-domain convolution involves complex folding and shifting operations, 
    the **frequency domain** offers a more intuitive alternative viewpoint.
    
    The **Discrete-Time Fourier Transform (DTFT)** and its computational approximation, 
    the **Fast Fourier Transform (FFT)**, convert a signal from the time domain to the frequency domain:
    """)
    
    st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}")
    
    st.markdown(r"""
    The result X[k] shows the amplitude and phase of each frequency component present in the signal. 
    Similarly, H[k] = FFT(h[n]) is called the **frequency response** of the system.
    
    ### The Convolution-Multiplication Property
    
    A remarkable and powerful property of the Fourier Transform is:
    """)
    
    st.latex(r"\text{Convolution in time domain} \quad \Leftrightarrow \quad \text{Multiplication in frequency domain}")
    
    st.markdown(r"""
    Mathematically:
    """)
    
    st.latex(r"y[n] = x[n] \circledast h[n] \quad \Leftrightarrow \quad Y[k] = X[k] \cdot H[k]")
    
    st.markdown(r"""
    This means that instead of computing the convolution sum (which requires N^2 operations for signals of length N), 
    we can:
    1. Transform x[n] and h[n] to the frequency domain using FFT (O(N log N) operations)
    2. Multiply the spectra pointwise (O(N) operations)
    3. Transform back to the time domain using inverse FFT (O(N log N) operations)
    
    This is vastly more efficient for long signals!
    
    ### The Comb Filter Effect
    
    In a room with periodic echoes, the impulse response is quasi-periodic. 
    Its frequency response H[k] exhibits a characteristic **comb filter** pattern:
    
    - **Peaks (constructive interference):** Occur at frequencies where the echo delays result in in-phase addition
    - **Nulls (destructive interference):** Occur at frequencies where echoes are out of phase and cancel
    
    The frequency spacing between peaks (and nulls) is inversely proportional to the echo delay:
    """)
    
    st.latex(r"\Delta f = \frac{F_s}{D}")
    
    st.markdown(r"""
    where D is the delay in samples.
    
    This is why convolved signals often sound "colored" or "filtered" - the room is selectively 
    amplifying and attenuating different frequencies based on its impulse response. 
    This is the physical basis of room acoustics!
    """)
    
    st.divider()
    
    # --- SUMMARY ---
    st.markdown("## Summary: The Complete Picture")
    st.markdown(r"""
    | Concept | Time Domain | Frequency Domain |
    |---------|------------|-----------------|
    | **Signal** | x[n] (sequence) | X[k] (spectrum) |
    | **System** | h[n] (impulse response) | H[k] (frequency response) |
    | **Output** | y[n] = x[n] * h[n] | Y[k] = X[k] * H[k] |
    | **Operation** | Convolution (folding, shifting, summing) | Pointwise multiplication |
    | **Complexity** | O(N^2) | O(N log N) (with FFT) |
    """)
    
    st.markdown(r"""
    ### Key Takeaways
    
    1. **Sampling** converts continuous audio into a discrete sequence x[n] at rate F_s.
    
    2. A **room** (or any acoustic space) is an **LSI system** characterized by its impulse response h[n].
    
    3. **Convolution** in the time domain computes the output: y[n] = x[n] * h[n].
    
    4. The **FFT** reveals the frequency-domain behavior and enables efficient computation: Y[k] = X[k] * H[k].
    
    5. The **comb filter effect** from periodic echoes creates peaks and nulls in the frequency response, 
       physically altering the perceived color of the sound.
    
    ### Educational Value
    
    This interactive tool lets you **experiment hands-on** with these concepts:
    - Adjust echo delays and attenuation to control h[n] in real-time
    - Observe how different impulse responses produce different outputs
    - Visualize both time-domain convolution and frequency-domain filtering
    - Listen to the acoustic effects and relate them to the mathematical models
    
    By exploring the interplay between time and frequency domains, you develop 
    intuition for why DSP is so powerful and pervasive in modern technology!
    """)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
    <p><strong>LSI Acoustic Studio</strong> -- A university-level educational platform</p>
    <p>Built with Streamlit, NumPy, SciPy, and Matplotlib</p>
</div>
""", unsafe_allow_html=True)
"""
LSI Acoustic Studio - Interactive DSP Learning Platform
A university-level educational tool for Signals and Systems concepts
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from scipy.fft import fft, fftfreq
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LSI Acoustic Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #2ca02c;
        margin-top: 1.5rem;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #d62728;
    }
    .metric-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_audio_from_mic():
    """Record audio from microphone using st.audio_input."""
    audio_data = st.audio_input("🎤 Record your voice")
    if audio_data is not None:
        try:
            audio_bytes = audio_data.getvalue()
            data, sr = sf.read(io.BytesIO(audio_bytes))
            return data, sr
        except Exception as e:
            st.sidebar.error(f"Error reading microphone input: {e}")
            return None, None
    return None, None


def load_audio_from_file(uploaded_file):
    """Load audio from uploaded .wav file."""
    if uploaded_file is not None:
        try:
            data, sr = sf.read(uploaded_file)
            return data, sr
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            return None, None
    return None, None


def generate_default_audio(duration: float = 2.0, sr: int = 16000):
    """Generate a synthetic test signal combining a chirp and noise burst."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Frequency sweep from 200 Hz to 2000 Hz
    f_start, f_end = 200, 2000
    chirp_signal = signal.chirp(t, f_start, duration, f_end, method='linear')
    
    # Add a noise burst in the middle (0.5 to 1.0 seconds)
    noise_burst = np.zeros_like(t)
    noise_start = int(0.5 * sr)
    noise_end = int(1.0 * sr)
    noise_burst[noise_start:noise_end] = np.random.normal(0, 0.3, noise_end - noise_start)
    
    # Combine signals
    combined_signal = 0.7 * chirp_signal + 0.3 * noise_burst
    
    return combined_signal, sr


def preprocess_audio(audio_data: np.ndarray, sr: int):
    """Normalize audio to [-1, 1] and convert stereo to mono."""
    # Convert stereo to mono by averaging channels
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)
    
    # Ensure 1D array
    audio_data = np.atleast_1d(audio_data.flatten())
    
    # Normalize to [-1, 1]
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    return audio_data, sr


def generate_echo_impulse_response(delay: float, attenuation: float, sr: int):
    """
    Generate impulse response for echo effect (Math Sandbox mode).
    h[n] = δ[n] + α δ[n-D] + α² δ[n-2D] + ...
    """
    delay_samples = int(delay * sr)
    max_samples = delay_samples * 4
    h = np.zeros(max_samples)
    
    h[0] = 1.0  # Primary impulse
    
    # Add 3 echoes with exponential decay
    for echo_idx in range(1, 4):
        echo_pos = echo_idx * delay_samples
        if echo_pos < len(h):
            h[echo_pos] = attenuation ** echo_idx
    
    return h


def generate_room_impulse_response(room_type: str, sr: int, duration: float = 0.5):
    """
    Simulate room impulse response for Real-World Vault mode.
    Generates different characteristics for Small Room, Large Hall, and Deep Cave.
    """
    num_samples = int(duration * sr)
    h = np.zeros(num_samples)
    
    if room_type == "Small Room":
        # Short, quick decay with minimal reflections
        decay_envelope = np.exp(-np.linspace(0, 5, num_samples))
        early_reflections = np.random.normal(0, 0.1, num_samples)
        h = decay_envelope * (0.8 + 0.2 * early_reflections)
        h[0] = 1.0
        
    elif room_type == "Large Hall":
        # Longer decay with structured early reflections
        decay_envelope = np.exp(-np.linspace(0, 8, num_samples))
        reflections = np.zeros(num_samples)
        # Add discrete early reflections at specific delays
        reflection_indices = [
            int(0.02 * sr), int(0.05 * sr), 
            int(0.08 * sr), int(0.12 * sr)
        ]
        for idx in reflection_indices:
            if idx < num_samples:
                reflections[idx] = 0.5 + np.random.normal(0, 0.1)
        h = decay_envelope * (0.6 + 0.4 * reflections)
        h[0] = 1.0
        
    elif room_type == "Deep Cave":
        # Highly diffuse response with very long tail
        decay_envelope = np.exp(-np.linspace(0, 10, num_samples))
        diffuse_reflections = np.random.normal(0.2, 0.15, num_samples)
        h = decay_envelope * np.abs(diffuse_reflections)
        h[0] = 1.0
    
    # Normalize
    h = h / (np.max(np.abs(h)) + 1e-10)
    return h


def convolve_signals(x: np.ndarray, h: np.ndarray):
    """Perform linear convolution using scipy.signal.convolve."""
    return signal.convolve(x, h, mode='full')


def compute_magnitude_spectrum(signal_data: np.ndarray, sr: int):
    """
    Compute magnitude spectrum using FFT.
    Returns frequencies and magnitudes up to 8000 Hz.
    """
    # Compute FFT
    fft_result = fft(signal_data)
    magnitude = np.abs(fft_result)
    
    # Compute frequency axis
    freqs = fftfreq(len(signal_data), 1/sr)
    
    # Keep only positive frequencies up to 8000 Hz
    positive_freq_mask = (freqs >= 0) & (freqs <= 8000)
    
    return freqs[positive_freq_mask], magnitude[positive_freq_mask]


# ============================================================================
# SIDEBAR: AUDIO INPUT & MODE SELECTION
# ============================================================================

st.sidebar.markdown("# ⚙️ Configuration Panel")

st.sidebar.markdown("## 🎵 Audio Input")

# Audio input method selection
audio_input_method = st.sidebar.radio(
    "Choose input signal x[n]:",
    ["🎤 Microphone Recording", "📁 Upload File", "🔧 Default Test Signal"],
    help="Select how to provide the input audio signal"
)

x = None
Fs = None
audio_loaded = False

if audio_input_method == "🎤 Microphone Recording":
    x, Fs = record_audio_from_mic()
    if x is not None:
        audio_loaded = True
    else:
        st.sidebar.info("No recording captured yet. Please record audio.")

elif audio_input_method == "📁 Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a .wav file", 
        type=["wav"],
        help="Select a WAV file from your computer"
    )
    if uploaded_file:
        x, Fs = load_audio_from_file(uploaded_file)
        if x is not None:
            audio_loaded = True

else:  # Default Test Signal
    if st.sidebar.button("🎛️ Generate Default Test Signal", width='stretch'):
        x, Fs = generate_default_audio()
        audio_loaded = True
        st.sidebar.success("✅ Test signal generated!")

# Preprocess audio if available
if audio_loaded and x is not None and Fs is not None:
    x, Fs = preprocess_audio(x, Fs)
    st.sidebar.success(f"✅ Audio loaded: {len(x)} samples @ {Fs} Hz")

st.sidebar.divider()

# Application mode selection
st.sidebar.markdown("## ⚙️ LSI System Mode")

mode = st.sidebar.radio(
    "Select System Mode:",
    ["Mode A: Math Sandbox", "Mode B: Real-World Vault"],
    help="Choose how to generate the impulse response"
)

h = None

if mode == "Mode A: Math Sandbox":
    st.sidebar.markdown("### Echo Parameters")
    
    delay = st.sidebar.slider(
        "Echo Delay (D) in seconds",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Time delay between echoes"
    )
    
    attenuation = st.sidebar.slider(
        "Attenuation (Alpha) α",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Decay factor per echo (0 < α < 1)"
    )
    
    if audio_loaded and x is not None and Fs is not None:
        h = generate_echo_impulse_response(delay, attenuation, Fs)
        st.sidebar.info(f"IR generated: {len(h)} samples ({len(h)/Fs:.3f} sec)")

else:  # Mode B: Real-World Vault
    st.sidebar.markdown("### Room Configuration")
    
    room_type = st.sidebar.selectbox(
        "Select Room Type:",
        ["Small Room", "Large Hall", "Deep Cave"],
        help="Choose an acoustic environment"
    )
    
    if audio_loaded and x is not None and Fs is not None:
        h = generate_room_impulse_response(room_type, Fs)
        st.sidebar.info(f"IR generated: {len(h)} samples ({len(h)/Fs:.3f} sec)")

st.sidebar.divider()

# Convolve button
process_audio = st.sidebar.button(
    "🔊 Convolve!", 
    width='stretch',
    type="primary",
    help="Click to compute convolution and update visualizations"
)

# ============================================================================
# MAIN CONTENT: TABS
# ============================================================================

st.title("🎵 LSI Acoustic Studio")
st.markdown("**An Interactive DSP Learning Platform for Signals and Systems**")
st.divider()

# Check if audio and impulse response are available
if x is None or Fs is None:
    st.warning("⚠️ **No audio signal loaded.** Please provide an input signal using the sidebar!")
    st.stop()

if h is None:
    st.warning("⚠️ **No impulse response generated.** Please select a system mode in the sidebar!")
    st.stop()

# Create tabs
tab1, tab2 = st.tabs(["🎛️ Acoustic Simulator", "📚 Signals & Systems Theory"])

# ============================================================================
# TAB 1: ACOUSTIC SIMULATOR
# ============================================================================

with tab1:
    st.header("Acoustic Simulator - Time and Frequency Domain Analysis")
    
    if process_audio:
        # Perform linear convolution
        y = convolve_signals(x, h)
        
        # Time axes
        t_x = np.arange(len(x)) / Fs
        t_h = np.arange(len(h)) / Fs
        t_y = np.arange(len(y)) / Fs
        
        # Create two-column layout for time and frequency domains
        col1, col2 = st.columns(2)
        
        # --- TIME DOMAIN ANALYSIS ---
        with col1:
            st.subheader("📊 Time Domain Analysis")
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 11))
            fig.patch.set_facecolor('white')
            
            # Plot 1: Input Signal
            axes[0].plot(t_x, x, linewidth=1.2, color='#1f77b4', alpha=0.9)
            axes[0].fill_between(t_x, x, alpha=0.2, color='#1f77b4')
            axes[0].set_xlabel('Time (seconds)', fontsize=10)
            axes[0].set_ylabel('Amplitude', fontsize=10)
            axes[0].set_title('Input Signal x[n]', fontsize=11, fontweight='bold')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].set_xlim([0, t_x[-1]])
            axes[0].axhline(y=0, color='k', linewidth=0.5)
            
            # Plot 2: Impulse Response (stem plot)
            stem_container = axes[1].stem(t_h, h, basefmt=' ')
            stem_container.markerline.set_color('#2ca02c')
            stem_container.markerline.set_markersize(8)
            stem_container.stemlines.set_color('#2ca02c')
            stem_container.stemlines.set_linewidth(1.5)
            axes[1].set_xlabel('Time (seconds)', fontsize=10)
            axes[1].set_ylabel('Amplitude', fontsize=10)
            axes[1].set_title('Impulse Response h[n]', fontsize=11, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].set_xlim([0, min(t_h[-1], 1.0)])
            axes[1].axhline(y=0, color='k', linewidth=0.5)
            
            # Plot 3: Output Signal
            axes[2].plot(t_y, y, linewidth=1.2, color='#d62728', alpha=0.9)
            axes[2].fill_between(t_y, y, alpha=0.2, color='#d62728')
            axes[2].set_xlabel('Time (seconds)', fontsize=10)
            axes[2].set_ylabel('Amplitude', fontsize=10)
            axes[2].set_title('Output Signal y[n] = x[n] * h[n]', fontsize=11, fontweight='bold')
            axes[2].grid(True, alpha=0.3, linestyle='--')
            axes[2].set_xlim([0, t_y[-1]])
            axes[2].axhline(y=0, color='k', linewidth=0.5)
            
            plt.tight_layout()
            st.pyplot(fig, width='stretch')
        
        # --- FREQUENCY DOMAIN ANALYSIS ---
        with col2:
            st.subheader("📈 Frequency Domain Analysis")
            
            # Compute magnitude spectra
            freq_x, mag_x = compute_magnitude_spectrum(x, Fs)
            freq_y, mag_y = compute_magnitude_spectrum(y, Fs)
            
            # Convert to dB scale
            mag_x_db = 20 * np.log10(mag_x + 1e-10)
            mag_y_db = 20 * np.log10(mag_y + 1e-10)
            
            # Plot overlaid spectra
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')
            
            ax.plot(freq_x, mag_x_db, label='Input x[n]', 
                   linewidth=1.8, color='#1f77b4', alpha=0.8)
            ax.plot(freq_y, mag_y_db, label='Output y[n]', 
                   linewidth=1.8, color='#d62728', alpha=0.8)
            
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('Magnitude (dB)', fontsize=10)
            ax.set_title('Magnitude Spectrum: Comb Filter Effect', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=10, loc='upper right')
            ax.set_xlim([0, 8000])
            
            plt.tight_layout()
            st.pyplot(fig, width='stretch')
        
        # --- AUDIO PLAYBACK ---
        st.divider()
        st.subheader("🔊 Audio Playback")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Signal**")
            st.audio(x, sample_rate=Fs, format='audio/wav')
            st.caption(f"Duration: {len(x)/Fs:.2f} sec")
        
        with col2:
            st.write("**Convolved Signal**")
            y_normalized = y / (np.max(np.abs(y)) + 1e-10)
            st.audio(y_normalized, sample_rate=Fs, format='audio/wav')
            st.caption(f"Duration: {len(y)/Fs:.2f} sec")
        
        # --- SYSTEM INFORMATION ---
        st.divider()
        st.subheader("📋 System Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sampling Rate", f"{Fs:,} Hz")
        with col2:
            st.metric("Input Length", f"{len(x):,} samples")
        with col3:
            st.metric("IR Length", f"{len(h):,} samples")
        with col4:
            st.metric("Output Length", f"{len(y):,} samples")
        
        # Display convolution formula
        st.divider()
        st.markdown("**Convolution Computation:**")
        st.latex(r"y[n] = \sum_{k=0}^{M-1} x[k] \cdot h[n-k]")
        st.markdown(f"where input length = {len(x)}, IR length = {len(h)}, "
                   f"output length = {len(y)}")
    
    else:
        st.info("👈 Configure the system in the sidebar and click **'Convolve!'** to start the analysis.")


# ============================================================================
# TAB 2: SIGNALS & SYSTEMS THEORY
# ============================================================================

with tab2:
    st.header("📚 Signals & Systems Theory")
    st.markdown("""
    This section provides the mathematical foundation for understanding the acoustic 
    simulator and the DSP concepts it demonstrates.
    """)
    
    # --- SECTION 1 ---
    st.markdown("## 1. Audio as a Discrete-Time Signal")
    st.markdown("""
    Continuous audio—sound pressure waves in air—is converted into a discrete-time signal 
    through a process called **sampling**. At a fixed sampling rate $F_s$ (typically 16,000 Hz 
    for telephony, 44,100 Hz for CD-quality audio), we measure the air pressure amplitude 
    at regular time intervals $T = 1/F_s$:
    """)
    
    st.latex(r"x[n] = x_c(nT) \quad \text{where} \quad n = 0, 1, 2, \ldots \quad \text{and} \quad T = \frac{1}{F_s}")
    
    st.markdown("""
    The resulting discrete sequence $x[n]$ is a sequence of real numbers, each representing 
    the amplitude of the audio signal at a specific point in time.
    
    **The Nyquist-Shannon Sampling Theorem** guarantees that if we sample a continuous signal 
    at rate $F_s$, we can faithfully reconstruct frequencies up to the **Nyquist frequency**:
    """)
    
    st.latex(r"f_{\text{Nyquist}} = \frac{F_s}{2}")
    
    st.markdown("""
    Any frequency component higher than $f_{\text{Nyquist}}$ will be aliased (incorrectly 
    represented) in the discrete signal. This is why microphone inputs are filtered before 
    sampling to remove high-frequency components.
    """)
    
    st.divider()
    
    # --- SECTION 2 ---
    st.markdown("## 2. The Room as a Linear Shift-Invariant (LSI) System")
    st.markdown("""
    A physical room behaves as a **Linear Shift-Invariant (LSI) system**. This means:
    
    **Linearity:** The system respects superposition. If input $x_1[n]$ produces output $y_1[n]$ 
    and input $x_2[n]$ produces output $y_2[n]$, then:
    """)
    
    st.latex(r"ax_1[n] + bx_2[n] \quad \rightarrow \quad ay_1[n] + by_2[n]")
    
    st.markdown("""
    **Time-Invariance (Shift-Invariance):** The system's behavior does not change over time. 
    If input $x[n]$ produces output $y[n]$, then a delayed input $x[n - n_0]$ produces a 
    proportionally delayed output:
    """)
    
    st.latex(r"x[n - n_0] \quad \rightarrow \quad y[n - n_0]")
    
    st.markdown(r"""
    The complete behavior of an LSI system is fully captured by a single function: 
    its **Impulse Response** $h[n]$. This is the output of the system when the input is 
    the **Dirac delta function** $\delta[n]$:
    """)
    
    st.latex(r"\delta[n] = \begin{cases} 1 & \text{if } n = 0 \\ 0 & \text{if } n \neq 0 \end{cases}")
    
    st.markdown("""
    In a room, the impulse response captures the primary sound plus all echoes from reflections:
    """)
    
    st.latex(r"h[n] = \delta[n] + \alpha \delta[n - D] + \alpha^2 \delta[n - 2D] + \alpha^3 \delta[n - 3D] + \cdots")
    
    st.markdown(r"""
    where:
    - $D$ is the delay (in samples) corresponding to the distance a sound wave travels before the first echo
    - $\alpha$ (with $0 < \alpha < 1$) is the **attenuation factor**, representing how much energy is lost per reflection
    
    For example, if the room is 34 meters long and sound travels at 340 m/s with $F_s = 16{,}000$ Hz, 
    then $D = 34 / 340 \times 16{,}000 = 1{,}600$ samples.
    """)
    
    st.divider()
    
    # --- SECTION 3 ---
    st.markdown("## 3. The Convolution Sum")
    st.markdown("""
    Given an input signal $x[n]$ and the impulse response $h[n]$ of an LSI system, 
    the output signal $y[n]$ is computed via the **discrete-time convolution sum**:
    """)
    
    st.latex(r"y[n] = \sum_{k=-\infty}^{\infty} x[k] \, h[n - k] = \sum_{k=-\infty}^{\infty} x[n-k] \, h[k]")
    
    st.markdown("""
    Conceptually, this operation consists of four steps:
    
    1. **Folding:** Reverse the impulse response in time to create $h[-k]$.
    2. **Shifting:** Shift the folded impulse response by $n$ positions to align with the current time.
    3. **Multiplication:** Multiply the shifted impulse response element-wise with the input signal.
    4. **Summation:** Sum all products to produce the single output sample $y[n]$.
    
    This process is repeated for every output index $n$, producing the complete output sequence $y[n]$.
    
    ### Example: Echo with Attenuation
    
    For a simple echo system where $h[n] = \\delta[n] + 0.5 \\delta[n - D]$, 
    the convolution produces:
    """)
    
    st.latex(r"y[n] = x[n] + 0.5 \, x[n - D]")
    
    st.markdown("""
    This is exactly what we expect: the original signal plus an attenuated copy delayed by $D$ samples.
    
    **Importance:** Convolution is the fundamental operation in signal processing. 
    Once you know $h[n]$, you can compute the system's output for *any* input signal via convolution. 
    No matter how complex the input, the convolution operation fully describes the system's response.
    """)
    
    st.divider()
    
    # --- SECTION 4 ---
    st.markdown("## 4. Frequency Domain Characterization")
    st.markdown(r"""
    While time-domain convolution involves complex folding and shifting operations, 
    the **frequency domain** offers a more intuitive alternative viewpoint.
    
    The **Discrete-Time Fourier Transform (DTFT)** and its computational approximation, 
    the **Fast Fourier Transform (FFT)**, convert a signal from the time domain to the frequency domain:
    """)
    
    st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j 2\pi k n / N}")
    
    st.markdown("""
    The result $X[k]$ shows the amplitude and phase of each frequency component present in the signal. 
    Similarly, $H[k] = \\text{FFT}(h[n])$ is called the **frequency response** of the system.
    
    ### The Convolution-Multiplication Property
    
    A remarkable and powerful property of the Fourier Transform is:
    """)
    
    st.latex(r"\text{Convolution in time domain} \quad \Leftrightarrow \quad \text{Multiplication in frequency domain}")
    
    st.markdown(r"""
    Mathematically:
    """)
    
    st.latex(r"y[n] = x[n] \circledast h[n] \quad \Leftrightarrow \quad Y[k] = X[k] \cdot H[k]")
    
    st.markdown(r"""
    This means that instead of computing the convolution sum (which requires $N^2$ operations for signals of length $N$), 
    we can:
    1. Transform $x[n]$ and $h[n]$ to the frequency domain using FFT ($O(N \log N)$ operations)
    2. Multiply the spectra pointwise ($O(N)$ operations)
    3. Transform back to the time domain using inverse FFT ($O(N \log N)$ operations)
    
    This is vastly more efficient for long signals!
    
    ### The Comb Filter Effect
    
    In a room with periodic echoes, the impulse response is quasi-periodic. 
    Its frequency response $H[k]$ exhibits a characteristic **comb filter** pattern:
    
    - **Peaks (constructive interference):** Occur at frequencies where the echo delays result in in-phase addition
    - **Nulls (destructive interference):** Occur at frequencies where echoes are out of phase and cancel
    
    The frequency spacing between peaks (and nulls) is inversely proportional to the echo delay:
    """)
    
    st.latex(r"\Delta f = \frac{F_s}{D}")
    
    st.markdown(r"""
    where $D$ is the delay in samples.
    
    This is why convolved signals often sound "colored" or "filtered"—the room is selectively 
    amplifying and attenuating different frequencies based on its impulse response. 
    This is the physical basis of room acoustics!
    """)
    
    st.divider()
    
    # --- SUMMARY ---
    st.markdown("## Summary: The Complete Picture")
    st.markdown(r"""
    | Concept | Time Domain | Frequency Domain |
    |---------|------------|-----------------||
    | **Signal** | $x[n]$ (sequence) | $X[k]$ (spectrum) |
    | **System** | $h[n]$ (impulse response) | $H[k]$ (frequency response) |
    | **Output** | $y[n] = x[n] \circledast h[n]$ | $Y[k] = X[k] \cdot H[k]$ |
    | **Operation** | Convolution (folding, shifting, summing) | Pointwise multiplication |
    | **Complexity** | $O(N^2)$ | $O(N \log N)$ (with FFT) |
    """)
    
    st.markdown(r"""
    ### Key Takeaways
    
    1. **Sampling** converts continuous audio into a discrete sequence $x[n]$ at rate $F_s$.
    
    2. A **room** (or any acoustic space) is an **LSI system** characterized by its impulse response $h[n]$.
    
    3. **Convolution** in the time domain computes the output: $y[n] = x[n] \circledast h[n]$.
    
    4. The **FFT** reveals the frequency-domain behavior and enables efficient computation: $Y[k] = X[k] \cdot H[k]$.
    
    5. The **comb filter effect** from periodic echoes creates peaks and nulls in the frequency response, 
       physically altering the perceived color of the sound.
    
    ### Educational Value
    
    This interactive tool lets you **experiment hands-on** with these concepts:
    - Adjust echo delays and attenuation to control $h[n]$ in real-time
    - Observe how different impulse responses produce different outputs
    - Visualize both time-domain convolution and frequency-domain filtering
    - Listen to the acoustic effects and relate them to the mathematical models
    
    By exploring the interplay between time and frequency domains, you develop 
    intuition for why DSP is so powerful and pervasive in modern technology!
    """)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
    <p><strong>LSI Acoustic Studio</strong> — A university-level educational platform</p>
    <p>Built with Streamlit, NumPy, SciPy, and Matplotlib</p>
</div>
""", unsafe_allow_html=True)
