# AFM Change Detection Model

This repository contains the core model definition for the manuscript:

**Efficient Remote Sensing Change Detection with Adaptive Frequency Masking for Pseudo-Change Suppression**


## Model

The implemented model takes a bi-temporal RGB image pair and predicts binary change logits.

```text
T1:     [B, 3, H, W]
T2:     [B, 3, H, W]
Output: [B, 1, H, W]
