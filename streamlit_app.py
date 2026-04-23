#!/usr/bin/env python3
"""
LSI Acoustic Studio - Vercel-compatible entry point
This is a lightweight wrapper for Vercel deployment.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Start the Streamlit app for Vercel"""
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'true'
    
    # Import and run streamlit
    from streamlit.cli import main as streamlit_main
    
    sys.argv = ['streamlit', 'run', 'app.py', '--server.port=8501', '--logger.level=warning']
    streamlit_main()

if __name__ == '__main__':
    main()
