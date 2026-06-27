# Edge-Computed Forest Fire Detection
**Phase 1: Resource-Optimized Vision Node Prototype**

---

## 📄 Detailed Project Report
For a comprehensive breakdown of the mathematical heuristics, architectural design, and performance metrics, please refer to the official research report:
**[View the Full Project Report Here](https://drive.google.com/file/d/1XBvG-Kr9XRvNs8rrVreVFde7z1D1aQHw/view?usp=sharing)**

---

## 📖 Research Overview

This project presents a **zero-latency, computationally lightweight edge-vision node** capable of deterministically filtering out visual noise for real-time forest fire detection.

Unlike conventional cloud-based AI solutions or standard optical cameras that are inherently vulnerable to environmental noise, this prototype utilizes a **deterministic, multi-layered mathematical pipeline**. By operating directly on the "edge" (the sensor node) and avoiding continuous neural network inference, the system prevents rapid thermal throttling and severe battery degradation, making it highly suitable for **unmanned aerial vehicles (drones)** or **remote IoT nodes**.

---

## 🧠 Research Methodology

To achieve a unique balance of speed and efficiency, the system utilizes a **three-stage heuristic filter**:

### 1. Spectral Analysis (Illumination Resilience)
Fuses **normalized RGB chromaticity** with **HSV color space** tracking. By utilizing programmatic hardware exposure attenuation (acting as "digital sunglasses"), the system prevents optical sensor saturation and accurately tracks plasma signatures in both direct sunlight and pitch-dark environments.

### 2. Morphological Gatekeeping (Geometric Analysis)
Employs algebraic geometry to differentiate organic fire from structured objects. It evaluates two primary metrics:

- **Boundary Complexity (Roughness):** Mathematically evaluates the chaotic, jagged boundaries of a fire by comparing the perimeter to its total area.
- **Shape Density (Solidity):** Calculates the ratio of the contour area to its convex hull to verify the presence of internal gaps and voids, instantly rejecting solid blocks of color like high-visibility clothing or vehicles.

### 3. Temporal Verification
Implements a strict **state-machine buffer of 15 contiguous frames** (approx. 0.5 seconds at 30 FPS) to eliminate transient noise such as single-frame digital sensor glitches or sunlight reflecting off windshields.

---

## ⚙️ Technical Specifications & Performance

| Specification | Detail |
|---|---|
| **Core Logic** | Deterministic Heuristic Filtering (eschews heavy CNN/DL overhead) |
| **Computational Efficiency** | Mathematically discards 99% of empty video frames in fractions of a millisecond |
| **Language** | Python 3.x |

### Primary Software Stack

- **OpenCV (`cv2`):** Optimized C++ video buffer manipulation.
- **NumPy:** Vectorized matrix operations for instantaneous high-frame-rate processing.
- **Pygame:** Asynchronous hardware actuation ensuring the critical video processing loop is not blocked by audio playback.

---

## 🚀 Local Installation & Execution

Follow these steps to set up and run the vision node on your local machine.

### Prerequisites

- **Python 3.8** or higher installed on your system.
- A working **webcam** connected to your computer.

### Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/RaviBaskey/edge-computed-forest-fire-detection.git
cd edge-computed-forest-fire-detection
```

### Step 2: Set Up a Virtual Environment (Recommended)

Isolating your dependencies ensures the project runs smoothly without conflicting with other Python packages on your system.

```bash
# For Windows:
python -m venv venv
venv\Scripts\activate

# For macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required libraries (`opencv-python`, `numpy`, `pygame`) using pip:

```bash
pip install -r requirements.txt
```

> [!TIP]
> If you do not have a `requirements.txt` file yet, you can install them manually:
> ```bash
> pip install opencv-python numpy pygame
> ```

### Step 4: Add Audio Asset

Ensure you have an audio file named `alarm-sound.mp3` placed in the **root directory** of the project. The system requires this file for asynchronous hardware actuation.

### Step 5: Execute the System

Run the core detection engine:

```bash
python fire-detector.py
```

The system will launch the **Hardware Live Feed** window.

> To exit the application, click on the video window and press the **`q`** key.

---

## 🔮 Future Scope (Phase 2)

The next phase of this research will transition the current deterministic pipeline into a **region-proposal generator**. Rather than running heavy AI on the entire video frame, the algorithm will crop a minuscule **64×64 pixel bounding box** around verified anomalies and forward it to an onboard, highly quantized **Convolutional Neural Network (CNN)** for final verification via **TinyML** integration.