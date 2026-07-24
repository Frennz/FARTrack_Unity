# FARTrack Unity Sentis

A Unity implementation of **FARTrack** (Fast Autoregressive Visual Tracking with High Performance) using **Unity Sentis**.

This repository contains a Unity project that runs FARTrack ONNX models directly on the GPU/CPU using Compute Shaders, without external dependencies like Python or OpenCV.

## Features

- **Unity Inference:** Uses `Unity.Sentis` for cross-platform inference.
- **Supported Models:** Supports `FARTrack-Tiny`, `FARTrack-Nano`, and `FARTrack-Pico`.
- **UI:** Includes a basic tracker UI for drawing bounding boxes with the mouse to start tracking.
- **Smoothing:** Includes Exponential Moving Average (EMA) coordinate smoothing and auto-loss detection heuristics.

---

## Getting Started

### 1. Requirements
- **Unity 2023.2 or newer**
- **Unity Sentis** (`com.unity.ai.inference` or `com.unity.sentis`) installed via the Package Manager.

### 2. Setup
1. Place your exported FARTrack `.onnx` model into the Unity `Assets/` folder. Unity Sentis will automatically import it as a Model Asset.
2. Add a `RawImage` to your UI Canvas to display a Webcam feed.
3. Attach the `FARTrackSentis` script to an empty GameObject.
4. Assign the references in the Inspector:
   - **Video Image:** Your RawImage UI element.
   - **Aspect Fitter:** (Optional) An AspectRatioFitter attached to your RawImage.
   - **Tracker Model:** Drag your imported `.sentis` model asset here.
   - **Coordinate Bins:** Set to `600` for Tiny/Nano/Pico models, or `4000` for the base/large models.

or just open the folder as project in unity.

### 3. Usage
Click Play in Unity. Drag a bounding box over an object in the webcam feed. The model will initialize and begin tracking the object in real-time.

---

## ONNX Models

This repository contains the Unity implementation. You can drop in pre-exported ONNX models. If you want to export the ONNX models yourself, you need to use the original FARTrack repository. 

**Note for exporting Nano and Pico models:**
If you export the Nano (10 layers) or Pico (6 layers) models, you must use the `FARTrackDistill` checkpoint provided by the original authors. The standard `FARTrack_tiny` checkpoint will not work for the shallower models because the intermediate layers were not trained to output coordinates.

---

## Acknowledgements

This is an Implemetation of FARTrack in Unity. All credit for the model architecture, training methodology, and original PyTorch implementation goes to the original authors.

**Original Repository:** [https://github.com/MIV-XJTU/FARTrack](https://github.com/MIV-XJTU/FARTrack)

Original Paper:
```bibtex
@inproceedings{wang2026fartrack,
  title={FARTrack: Fast Autoregressive Visual Tracking with High Performance},
  author={Wang, Guijie and others},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## License

This project is released under the **Apache 2.0 License**, inherited from the original FARTrack repository.
