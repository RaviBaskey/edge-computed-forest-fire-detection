# Edge-Computed Forest Fire Detection
**Phase 1: Resource-Optimized Vision Node Prototype**

## Research Overview
This project presents a zero-latency, heuristic-based computer vision system designed for real-time forest fire detection. Unlike conventional cloud-based AI solutions, this prototype utilizes a deterministic, multi-layered mathematical pipeline. By operating directly on the "edge" (the sensor node), the system eliminates network latency and drastically reduces power consumption, making it suitable for low-power IoT drone deployment.

## Research Methodology
This project addresses the critical challenge of false positives in visual fire detection using a three-stage heuristic filter:
1.  **Spectral Analysis:** Fuses RGB normalized chromaticity with HSV color space tracking to remain illumination-invariant across dynamic environments.
2.  **Morphological Gatekeeping:** Employs algebraic geometry (Boundary Complexity & Shape Density) to differentiate organic plasma structures from manufactured visual noise (e.g., flashlights, tubelights, or orange textiles).
3.  **Temporal Verification:** Implements a state-machine buffer to ensure anomaly persistence, effectively rejecting transient environmental glitches.

## Technical Specifications
* **Core Logic:** Deterministic Heuristic Filtering (avoids high-overhead Deep Learning inference).
* **Language:** Python 3.x
* **Primary Libraries:** * `OpenCV` (cv2): Optimized video buffer manipulation.
    * `NumPy`: Vectorized matrix operations for high-frame-rate performance.
    * `Pygame`: Asynchronous, non-blocking hardware alarm actuation.
* **Hardware Interfacing:** Direct exposure attenuation (digital "sunglasses") to preserve raw spectral fidelity in high-contrast/dark environments.

## Installation & Execution
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RaviBaskey/edge-computed-forest-fire-detection.git](https://github.com/RaviBaskey/edge-computed-forest-fire-detection.git)