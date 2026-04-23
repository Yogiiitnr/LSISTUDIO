# 🎵 LSI Acoustic Studio PRO

**Enterprise-Grade Digital Signal Processing Platform** | Built with Streamlit + SciPy

[![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-000000?style=flat&logo=vercel)](https://lsi-studio.vercel.app)
[![Python 3.13](https://img.shields.io/badge/python-3.13.7-blue)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28.1-ff0000)](https://streamlit.io)

## ✨ Features

- **6 Interactive Tabs** - Convolution, Filters, Venues, IIR Design, Analysis, Effects
- **8 World Venues** - Taj Mahal, Sydney Opera House, Pantheon, Colosseum, etc.
- **10 Music Filters** - Vocal Booster, Bass Enhancer, Treble Sparkle, Lo-Fi, etc.
- **4 Filter Types** - Butterworth, Chebyshev I/II, Elliptic
- **Real-Time Visualization** - Waveforms, Spectrograms, Frequency Response, Phase/Group Delay
- **4 Audio Effects** - Vibrato, Tremolo, Distortion, Echo
- **Dark Professional UI** - Purple-pink gradient theme

## 🚀 Quick Start

### Local Development
```bash
# Clone the repo
git clone https://github.com/yourusername/lsi-acoustic-studio.git
cd lsi-acoustic-studio

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py --server.port=8504
```

**Access at:** http://localhost:8504

### Deploy on Vercel
```bash
pip install vercel
vercel deploy
```

## 📊 Architecture

```
app.py (323 lines)
├── Sidebar: Audio Input (Mic/Upload/Generate)
├── Tab 1: Convolution Engine (Echo + Venues)
├── Tab 2: Music Filters (10 pre-designed)
├── Tab 3: World Venues (8 acoustic spaces)
├── Tab 4: IIR Filter Designer (4 types)
├── Tab 5: Analysis (Spectrogram/PSD)
└── Tab 6: Effects (Vibrato/Tremolo/Distortion/Echo)
```

## 🎯 Core Components

**Audio Input**
- Live microphone recording
- .wav file upload
- 5 test signals (Speech, Tone, Harmonic, Noise, Impulse)

**Signal Processing**
- FFT/IFFT analysis
- Convolution with custom/venue impulse responses
- IIR filter design & application
- Real-time frequency response visualization

**Visualization**
- Dark matplotlib theme (#0f0f1e, #1a1a2e)
- 5-plot convolution analysis
- Frequency domain comparisons
- Energy decay curves
- Group delay plots

## 🛠️ Technologies

| Component | Tech | Version |
|-----------|------|---------|
| Framework | Streamlit | 1.28.1 |
| DSP Core | NumPy + SciPy | 1.24.3 + 1.11.1 |
| Audio I/O | soundfile | 0.12.1 |
| Visualization | Matplotlib | 3.7.2 |
| Runtime | Python | 3.13.7 |

## ⚡ Key Features Explained

### Convolution Engine
- **Echo Mode**: Custom delay (50ms-1s), attenuation (10-95%), echoes (1-5)
- **Venue Mode**: 8 iconic world venues with realistic impulse responses
- **Pre-filtering**: Optional butterworth/cheby1 before convolution
- **Full Analysis**: Input, IR, output, frequency domain, signal lengths

### Filter Designer
- **Butterworth**: Maximally flat passband
- **Chebyshev I**: Ripple in passband
- **Chebyshev II**: Ripple in stopband
- **Elliptic**: Ripple in both bands (steepest roll-off)
- **Modes**: Lowpass, Highpass, Bandpass, Bandstop

### World Venues
- Taj Mahal (5.5s decay, 25 reflections)
- Sydney Opera House (3.5s, 18 reflections)
- Pantheon Rome (8.0s, 35 reflections)
- Grand Central (4.2s, 22 reflections)
- Ancient Pagoda (2.8s, 15 reflections)
- Church Santo Domingo (6.5s, 30 reflections)
- Bamboo Grove (1.5s, 8 reflections)
- Colosseum (7.2s, 40 reflections)

## 📝 Session State

```python
st.session_state.audio   # np.ndarray or None
st.session_state.sr      # int (sample rate, default 16000)
```

All tabs access via: `st.session_state.audio` and `st.session_state.sr`

## 🎨 Styling

**Color Palette**
- Primary: #667eea → #764ba2 → #f093fb
- Green: #43e97b | Red: #f5576c | Cyan: #4facfe
- Background: #0f0f1e (darkest) → #1a1a2e (dark)

**Typography**: Poppins (Google Fonts)

## 🔍 Critical Implementation Notes

### numpy Array Checks
```python
# ✅ Correct
if sos is None:  # Works with numpy arrays

# ❌ Wrong
if sos:  # Fails - "truth value ambiguous"
```

### Matplotlib Figures
```python
# ✅ Use constrained_layout
fig = plt.figure(..., constrained_layout=True)

# ❌ Don't use tight_layout
fig.tight_layout()  # Rendering issues
```

### Filter Cutoff Normalization
```python
# ✅ Normalize to Nyquist
nq = sr/2
c_norm = max(0.001, min(0.999, cutoff/nq))

# ❌ Don't pass raw cutoff
butter(order, cutoff_raw, ...)  # scipy errors
```

## 📦 Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application (323 lines) |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Streamlit configuration |
| `.gitignore` | Git ignore patterns |
| `README.md` | This file |

## 🚀 Deployment

### Vercel
```bash
vercel deploy --prod
```

**Environment Variables**: None required (fully client-side processing)

### GitHub
```bash
git add .
git commit -m "Initial commit: LSI Acoustic Studio PRO"
git branch -M main
git remote add origin https://github.com/yourusername/lsi-acoustic-studio.git
git push -u origin main
```

## ✅ Testing Checklist

- ✅ Sidebar: Mic, Upload, Generate
- ✅ Tab 1: Echo + Venue convolution
- ✅ Tab 2: All 10 filters
- ✅ Tab 3: All 8 venues
- ✅ Tab 4: All filter types/modes
- ✅ Tab 5: Spectrogram & PSD
- ✅ Tab 6: All 4 effects
- ✅ Zero runtime errors
- ✅ Dark theme rendering
- ✅ Audio playback

## 📖 Usage Examples

1. **Simulate Taj Mahal Acoustics**
   - Load audio → Tab 3 → Select "Taj Mahal" → Simulate

2. **Design Custom Filter**
   - Load audio → Tab 4 → Select butterworth lowpass @ 2kHz → Apply

3. **Apply Music Filter**
   - Load audio → Tab 2 → Select "Vocal Booster" → Apply

4. **Create Echo Effect**
   - Load audio → Tab 1 → Echo mode → Adjust delay/attenuation → Compute

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Audio not playing | Check sample rate matches (`st.session_state.sr`) |
| Graphs look similar | Use Venues or high attenuation echoes |
| Filter design fails | Cutoff must be normalized by Nyquist |
| Matplotlib rendering fail | Ensure `constrained_layout=True` |

## 📄 License

MIT License - Open source and free to use

## 👨‍💻 Author

Built as an enterprise-grade DSP platform for audio signal processing education and research.

---

**Status**: ✅ Production-Ready | **Errors**: 0 | **Lines**: 323
