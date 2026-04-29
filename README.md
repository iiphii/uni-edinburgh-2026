# Real-Time Neural Audio Models

University of Edinburgh, April 30, 2026

This repository contains code examples which demonstrate how to implement a UNet model for real-time audio processing.
The model is built step by step, starting by introducing the basic building blocks: Conv1d and ConvTranspose1d layers.
Then, we will make the model explicitly stateful, which makes it easy to reason about the model's behaviour in a real-time setting.
The goal is the make the training and low-latency inference code as similar as possible, so that the model can be easily deployed in a real-time setting after training.

## List of files

00_conv1d.py: A simple example of a Conv1d layer - showing its impulse response.

01_two_layers.py: A sequence of two Conv1d layers - demostrating how we lose causality with valid padding.
02_more_layers.py: A sequence of more Conv1d layers - just adding more layers.

03_dilations.py: A sequence of dilated Conv1d layers - showing how we can increase the receptive field without increasing the number of parameters.
04_more_dilations.py: A sequence of more dilated Conv1d layers - just adding more layers.

05_stateful.py: A stateful implementation of the model - showing the stateful implementation and how it makes the model causal.
06_forward_in_segments.py: Demonstrating how we can do forward passes with small buffers when using the stateful implementation.
07_stride.py: introducing the stride parameter in Conv1d layers - showing how we can downsample the signal and increase the receptive field.

08_transposed_conv1d.py: A simple example of a ConvTranspose1d layer - the final building block for our UNet model.

09_down_and_up.py: Unet model with a sequence of downsampling layers followed by a sequence of upsampling layers.
10_skip_connections.py: Adding skip connections to the UNet model and how they affect the impulse response.
11_deeper.py: just adding more layers

12_train.py: training a UNet model on a real audio dataset.

## Requirements
- pytorch
- numpy
- soundfile
- tqdm
- matplotlib
