# Real-Time Neural Audio Models

**University of Edinburgh**  
April 30, 2026

This repository contains code examples demonstrating how to implement a U-Net model for real-time audio processing.

The model is built step by step, starting with the basic building blocks: `Conv1d` and `ConvTranspose1d` layers.

From there, the model is made explicitly stateful, which makes it easier to reason about its behavior in a real-time setting.

The goal is to make the training and low-latency inference code as similar as possible, so the model can be easily deployed for real-time use after training.

---

## File Overview

### Basic Convolutions

- **00_conv1d.py**  
  A simple example of a `Conv1d` layer, showing its impulse response.

- **01_two_layers.py**  
  A sequence of two `Conv1d` layers, demonstrating how causality is lost with valid padding.

- **02_more_layers.py**  
  A deeper sequence of `Conv1d` layers with additional layers.

---

### Dilated Convolutions

- **03_dilations.py**  
  A sequence of dilated `Conv1d` layers, showing how to increase the receptive field without increasing the number of parameters.

- **04_more_dilations.py**  
  A deeper version with additional dilated layers.

---

### Stateful Processing

- **05_stateful.py**  
  A stateful implementation of the model, showing how statefulness makes the model causal.

- **06_forward_in_segments.py**  
  Demonstrates how to perform forward passes using small buffers with the stateful implementation.

- **07_stride.py**  
  Introduces the `stride` parameter in `Conv1d` layers, showing how to downsample the signal and further increase the receptive field.

---

### Upsampling and U-Net Construction

- **08_transposed_conv1d.py**  
  A simple example of a `ConvTranspose1d` layer — the final building block for the U-Net model.

- **09_down_and_up.py**  
  A U-Net model with a sequence of downsampling layers followed by a sequence of upsampling layers.

- **10_skip_connections.py**  
  Adds skip connections to the U-Net model and shows how they affect the impulse response.

- **11_deeper.py**  
  A deeper U-Net model with additional layers.

---

### Training

- **12_train.py**  
  Training a U-Net model on a real audio dataset.

---

## Requirements

- `pytorch`
- `numpy`
- `soundfile`
- `tqdm`
- `matplotlib`
