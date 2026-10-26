"""Runner script for PickleTrack-Local Web UI.

Launches the local interactive FastAPI server with Uvicorn.
Open http://localhost:8000 in your browser to use the graphical interface.
"""

import sys
import uvicorn

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    print("\n" + "=" * 65)
    print("    PICKLETRACK-LOCAL: INTERACTIVE VISION & ANALYTICS STUDIO   ")
    print("=" * 65)
    print(f"  * Web Dashboard URL : http://{host}:{port}")
    print("  * Features           : 4-Corner Canvas Calibration, Live HSV Tuner,")
    print("                         Range Scrubber, Video Playback & Shot Log CSV")
    print("=" * 65 + "\n")
    uvicorn.run("src.web.app:app", host=host, port=port, reload=False, log_level="info")

