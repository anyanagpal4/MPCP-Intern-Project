# Real-Time Eye Tracking System (MediaPipe + EAR)

This project detects **eye state (whether OPEN/CLOSED)** from a live webcam (or a video file) using:

- **OpenCV** for real-time video IO + drawing overlays
- **MediaPipe Face Mesh** for facial landmarks
- **Eye Aspect Ratio (EAR)** geometry for blink/eye-state classification

This follows the requirements from the `MPCP Intern Project: Real-Time Eye Tracking System` google document.

## Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r eye_tracking_project/requirements.txt
```

Verify you’re using the venv Python:

```bash
which python
python -V
```

## Run (Webcam)

```bash
python -m eye_tracking_project.eye_tracker --source 0 --width 640 --height 480 --threshold 0.21
```

- Press **`q`** to quit
- Press **`+`** / **`-`** to adjust the EAR threshold at runtime

### MediaPipe note (Solutions vs Tasks)

Some MediaPipe installs include the legacy **Solutions** API (`mp.solutions.face_mesh`) and will work out of the box.

Newer MediaPipe installs (like `mediapipe==0.10.32`) may only include the **Tasks** API and require a **FaceLandmarker model file** (`.task`).
If you see a message about missing `mp.solutions.face_mesh`, download a FaceLandmarker model and run with:

```bash
python -m eye_tracking_project.eye_tracker --source 0 --model /path/to/face_landmarker.task
```

#### How to download the `.task` model

- Go to the official **MediaPipe Face Landmarker** page and download the **FaceLandmarker** model (`.task`): [MediaPipe Face Landmarker](https://ai.google.dev/mediapipe/solutions/vision/face_landmarker)
- Alternatively, the MediaPipe Tasks hub also links to task models: [MediaPipe Tasks](https://mediapipe.dev/tasks/)

After downloading, place the file here (recommended), so you don’t have to pass `--model` every time:

- `eye_tracking_project/models/face_landmarker.task`

Then run:

```bash
python -m eye_tracking_project.eye_tracker --source 0
```

### If the webcam won’t open (on macOS)

If you see an error like “Could not open video source: 0”, try the following fixes:

- Grant camera permission to your Terminal / Python in **System Settings → Privacy & Security → Camera**
- Close other apps that might be using the camera (Zoom/Meet/etc.)
- Probe indices quickly: `--probe-cameras`
- Force AVFoundation backend:

```bash
python -m eye_tracking_project.eye_tracker --source 0 --backend avfoundation
```

- Or force default backend:

```bash
python -m eye_tracking_project.eye_tracker --source 0 --backend default
```

## Run (Video File)

```bash
python -m eye_tracking_project.eye_tracker --source /path/to/video.mp4
```

## Output Overlay

The window shows:

- Eye contour + landmark points
- **State** (OPEN/CLOSED/NO FACE)
- **EAR** value
- **Threshold**
- **Frame counter**
- **FPS**

## EAR Formula

For 6 eye landmarks $(p_1..p_6)$:

$$
EAR = \\frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \\cdot ||p_1 - p_4||}
$$

Classification:

- EAR < 0.21 → **CLOSED**
- EAR ≥ 0.21 → **OPEN**

## Landmark Indices (MediaPipe Face Mesh)

- Left Eye: `[362, 385, 387, 263, 373, 380]`
- Right Eye: `[33, 160, 158, 133, 153, 144]`

## Tests (Optional)

```bash
pytest -q
```

## Known Limitations

- Threshold may need tuning depending on camera angle/lighting and individual face geometry.
- “NO FACE” simply means the program didn’t detect a face in the current frame.

