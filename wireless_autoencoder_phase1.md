# Technical Reference Document

**Subject:** Mathematical and Code Specification of an End-to-End Wireless Autoencoder (Phase 1)

**Author:** AI Research Collaborator
**Date:** July 16, 2026

---

## 1. Executive Summary

This document provides a rigorous mathematical formulation and code analysis of an end-to-end communication system modeled as a Deep Neural Network (DNN) Autoencoder, inspired by the landmark work of O'Shea and Hoydis.

In classical communications, transceiver components (source coders, channel coders, modulators, and equalizers) are mathematically optimized in isolation. In contrast, the autoencoder framework treats the entire physical layer as a single, joint optimization problem. By optimizing an encoder (transmitter), a stochastic channel model, and a decoder (receiver) concurrently, the system naturally learns modulation geometries optimized for specific hardware limitations and channel environments.

---

## 2. Mathematical Foundations

The end-to-end communications chain is modeled as an unsupervised reconstruction pipeline operating over a stochastic channel.

```
[Message m] ---> ( Encoder f_θ ) ---> [s] ---> ( Power Norm ) ---> [x_norm]
                                                                        |
                                                                  (AWGN Channel)
                                                                        |
[Predicted m̂] <--- ( Decoder g_φ ) <-------------------------------- [r]
```

### 2.1 Message Representation Space

Let $M$ be the total number of unique messages the system can transmit (e.g., $M=16$, which represents $k = \log_2(M) = 4$ bits of information). A message $m \in \{1, 2, \dots, M\}$ is represented as a one-hot encoded vector $s \in \mathbb{R}^M$:

$$s = [s_1, s_2, \dots, s_M]^T$$

Where:

$$s_i = \begin{cases} 1 & \text{if } i = m \\ 0 & \text{if } i \neq m \end{cases}$$

This representation ensures that all messages are initially equidistant in the input vector space.

### 2.2 The Transmitter (Encoder) Function

The transmitter is defined as a parameterizable function $f_\theta: \mathbb{R}^M \to \mathbb{R}^n$, mapping the high-dimensional input vector $s$ to a low-dimensional physical channel space of dimension $n$:

$$x = f_\theta(s)$$

Where:

- $\theta$ represents the trainable weights and biases of the transmitter network.
- $n$ represents the number of real channel uses (dimensions). In a single-carrier complex system, $n = 2$, representing the In-Phase (I) and Quadrature (Q) components.

### 2.3 The Power Normalization Constraint

Physical transmitters are constrained by maximum or average power limitations. Without a constraint, the optimization process would drive $\lVert x \rVert$ to infinity to render channel noise negligible.

To enforce a strict average energy constraint of 1 per channel dimension, an L2-normalization is applied to the transmitter's output:

$$x_{norm} = \sqrt{n} \cdot \frac{x}{\lVert x \rVert_2} = \sqrt{n} \cdot \frac{x}{\sqrt{\sum_{i=1}^{n} x_i^2}}$$

This mathematical constraint projects the raw output vectors onto an $n$-dimensional hypersphere of radius $\sqrt{n}$. For $n = 2$, the outputs are restricted to a circle of radius $\sqrt{2}$.

### 2.4 The Channel Model (AWGN)

The physical channel is modeled as a non-trainable, stochastic Additive White Gaussian Noise (AWGN) layer. The received vector $r \in \mathbb{R}^n$ is defined as:

$$r = x_{norm} + n$$

Where the noise vector $n$ is sampled from a multivariate normal distribution:

$$n \sim \mathcal{N}(0, \sigma^2 I_n)$$

The noise variance $\sigma^2$ per dimension is computed using the energy per bit to noise power spectral density ratio ($E_b/N_0$) and the spectral efficiency of the system:

$$\sigma^2 = \frac{1}{2 \cdot \left(\frac{n}{k}\right) \cdot \left(\frac{E_b}{N_0}\right)_{linear}}$$

Where:

- $k = \log_2(M)$ is the number of bits per message.
- $\left(\frac{E_b}{N_0}\right)_{linear} = 10^{(E_b/N_0)_{dB}/10}$

### 2.5 The Receiver (Decoder) Function

The receiver is defined as a parameterizable function $g_\phi: \mathbb{R}^n \to \mathbb{R}^M$, where $\phi$ represents the weights and biases of the receiver network. It maps the corrupted, continuous channel symbols back into a probability distribution over the original $M$ possible classes:

$$z = g_\phi(r)$$
$$p = \text{softmax}(z)$$

The probability of the $i$-th message being the transmitted message is calculated via:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{M} e^{z_j}}$$

The ultimate hard-decision prediction $\hat{m}$ is extracted via the argmax operator:

$$\hat{m} = \arg\max_i p_i$$

### 2.6 Loss Optimization: Categorical Cross-Entropy

To optimize the joint parameters $\{\theta, \phi\}$, we minimize the Categorical Cross-Entropy loss over a training batch of size $B$. For a single sample where message $m$ was transmitted, the loss is formulated as:

$$L(\theta, \phi) = -\sum_{i=1}^{M} s_i \log(p_i) = -\log(p_m)$$

By propagating the gradient of this loss backward through the decoder, the non-trainable channel layer, and the encoder, the network dynamically updates its weights to maximize the Euclidean distance between constellation points under the L2 power constraint.

---

## 3. Code Walkthrough and Architecture

Below is the structured breakdown of the PyTorch script used to implement and evaluate Phase 1.

### 3.1 Network Class Definition

The following Python class defines the end-to-end model. It encapsulates the transmitter network, the normalization step, the channel, and the receiver network.

```python
import torch
import torch.nn as nn
import numpy as np

class WirelessAutoencoder(nn.Module):
    def __init__(self, M, n_channel_dims):
        super(WirelessAutoencoder, self).__init__()
        self.M = M
        self.n_dims = n_channel_dims
        
        # Transmitter (Encoder Network)
        # Input: 16-dimensional one-hot vector
        # Hidden layer: 16 units with ReLU activation
        # Output: 2-dimensional vector (representing raw I & Q)
        self.encoder = nn.Sequential(
            nn.Linear(self.M, self.M),
            nn.ReLU(),
            nn.Linear(self.M, self.n_dims) 
        )
        
        # Receiver (Decoder Network)
        # Input: 2-dimensional noisy channel symbol
        # Hidden layer: 16 units with ReLU activation
        # Output: 16-dimensional logit vector
        self.decoder = nn.Sequential(
            nn.Linear(self.n_dims, self.M),
            nn.ReLU(),
            nn.Linear(self.M, self.M)
        )
        
    def forward(self, x, noise_variance):
        # Step 1: Map the input index to unnormalized 2D coordinates
        s = self.encoder(x)
        
        # Step 2: Apply L2 normalization to enforce the average power constraint
        # s_norm has an average power of 1 per dimension
        s_norm = np.sqrt(self.n_dims) * (s / torch.norm(s, p=2, dim=1, keepdim=True))
        
        # Step 3: Simulate AWGN channel
        # Standard Normal distribution scaled by standard deviation (sqrt of variance)
        noise = torch.randn_like(s_norm) * np.sqrt(noise_variance)
        r = s_norm + noise
        
        # Step 4: Map received signal back to 16 output logits
        y = self.decoder(r)
        return y, s_norm
```

### 3.2 Analysis of the Normalization Step

```python
s_norm = np.sqrt(self.n_dims) * (s / torch.norm(s, p=2, dim=1, keepdim=True))
```

- `torch.norm(s, p=2, dim=1, keepdim=True)`: Calculates the Euclidean distance $\sqrt{I^2 + Q^2}$ for each batch sample independently.
- `s / torch.norm(...)`: Divides the raw outputs by their length, mapping every coordinate vector onto a unit circle (radius of 1).
- `np.sqrt(self.n_dims)`: Since the channel uses $n = 2$ dimensions, we scale the vector by $\sqrt{2}$. This step ensures that the total power distributed across both channel dimensions is $1 + 1 = 2$, achieving an average power of 1 per channel dimension.

---

## 4. Key Architectural Insights: Circular Geometry (16-PSK)

During training, the autoencoder was tasked with arranging 16 distinct message points inside a 2-dimensional space such that they remained as distinct as possible when corrupted by Gaussian noise.

### 16-QAM vs. 16-PSK

- **16-QAM (Quadrature Amplitude Modulation):** Traditional systems use a square 4×4 grid layout. While highly spectral-efficient, it requires points to be placed at different amplitude levels (some close to the origin, some far away). This creates a high Peak-to-Average Power Ratio (PAPR).
- **16-PSK (Phase Shift Keying):** The neural network converged on a circular constellation layout where all 16 points are distributed symmetrically around the boundary of the maximum power ring.

### Why the AI Selected a Circle

The loss optimizer (gradient descent) seeks the configuration that minimizes cross-entropy. Because of our strict average power normalization constraint, the most mathematically efficient way to maximize the separation between all 16 points is to push them to the furthest edge of the allowed energy boundary.

By distributing them evenly along the circumference of a circle of radius $\sqrt{2}$, the network independently derived 16-PSK. This geometry provides:

- **Equal Error Protection:** Since every point is equidistant from the origin and its adjacent neighbors, all messages share the same probability of error.
- **Optimal PAPR of 1:** Because the amplitude of every symbol is constant, hardware amplifiers can run at peak power without driving the signal into non-linear distortion regions.
