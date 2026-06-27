# CompositeAI: A Hybrid Classical and Deep Learning Framework for Realistic Image Insertion

## Overview

CompositeAI is an end-to-end image compositing framework designed to generate realistic image insertions by combining classical computer vision techniques with modern deep learning models.

The framework integrates:

* Foreground Extraction
* Object Placement
* Color Harmonization
* Shadow Synthesis
* Quality Evaluation
* Interactive Gradio Interface

The goal is to create visually convincing composite images where inserted objects naturally blend into target scenes through appearance adaptation and realistic shadow generation.

---

## Project Motivation

Traditional copy-and-paste image editing often produces unrealistic results because foreground objects and background scenes differ in:

* Illumination
* Color distribution
* Environmental context
* Shadow consistency
* Perspective alignment

CompositeAI addresses these challenges using a hybrid pipeline that combines user-controllable classical methods with state-of-the-art deep learning models.

---

## System Architecture

<p align="center">
  <img width="1084" height="372" alt="image" src="https://github.com/user-attachments/assets/ec1a8d52-1fec-43ac-a018-14c99e697d33" />
</p>

The framework consists of five major stages:

1. Foreground Extraction
2. Object Placement
3. Color Harmonization
4. Shadow Synthesis
5. Evaluation

---

## Key Contributions

✅ Hybrid Classical + Deep Learning Framework

✅ Multiple Foreground Extraction Models

✅ Multiple Harmonization Techniques

✅ GPSDiffusion-Based Shadow Synthesis

✅ Interactive Gradio Interface

✅ Evaluation Framework

✅ End-to-End Image Compositing Pipeline

✅ Unified Integration of All Components into a Single Workflow

---

## Methodology

### 1. Foreground Extraction

Classical Method:

* KNN Matting

Deep Learning Models:

* U²Net
* BRIA RMBG
* MODNet
* RobustVideoMatting
* ViTMatte

These models enable accurate extraction of complex structures including hair, clothing boundaries, and transparent regions.

---

### 2. Object Placement

The extracted object can be interactively positioned within a target scene using:

* X Position
* Y Position
* Scale Control

This enables realistic scene composition and perspective alignment.

---

### 3. Color Harmonization

#### Classical Methods

* Histogram Matching
* Reinhard Color Transfer
* LAB Color Transfer
* White Balance Adjustment

#### Deep Learning Methods

* PCTNet
* LBM
* Depth-Aware Harmonization

These techniques adapt the foreground appearance to match the target environment.

---

### 4. Shadow Synthesis

#### Geometric Shadow Projection

* Fast
* Lightweight
* CPU-Friendly

#### GPSDiffusion

* Scene-Aware Shadow Generation
* Deep Learning Based
* Context-Aware Lighting Adaptation
* Improved Shadow Realism

Based on recent advances in diffusion-based shadow generation.

---

## Interactive Gradio Interface

<p align="center">
  <img src="images/ui.png" width="900">
</p>

The system includes an interactive user interface that allows users to:

* Upload foreground and background images
* Select extraction models
* Apply harmonization techniques
* Generate realistic shadows
* Compare outputs
* Export final composite images

---

## Results

### Example 1

<p align="center">
<img width="236" height="334" alt="image" src="https://github.com/user-attachments/assets/debbb371-540f-4c92-8201-55d6d4b55501" />  <img width="274" height="334" alt="image" src="https://github.com/user-attachments/assets/c22242e8-eef1-4cbe-b970-6a55b5842a68" />   <img width="424" height="330" alt="image" src="https://github.com/user-attachments/assets/6e36ea41-4fa7-4692-a812-0fed8500a35f" />
</p>

### Example 2

<p align="center">
<img width="315" height="223" alt="image" src="https://github.com/user-attachments/assets/d8849932-a2b0-40ef-8097-aaed477306c2" /> <img width="369" height="238" alt="image" src="https://github.com/user-attachments/assets/9e6f65d5-36c6-472b-af7e-635d6becf387" /> <img width="356" height="245" alt="image" src="https://github.com/user-attachments/assets/abab3dad-4022-4fc4-815b-60ce8f3a168a" />
</p>

### Example 3

<p align="center">
A.)<img width="340" height="290" alt="image" src="https://github.com/user-attachments/assets/74f49e0f-709a-41ca-ba9a-843c0b8640c5" />     B.)<img width="330" height="284" alt="image" src="https://github.com/user-attachments/assets/5dbe2347-e4e1-49dc-80b5-54a46a3d0106" />

  <img width="801" height="156" alt="image" src="https://github.com/user-attachments/assets/c0dd28ff-6fe1-421f-ba68-7e067821a29a" />
</p>

The generated composites demonstrate:

* Improved color consistency
* Better illumination matching
* More realistic shadow placement
* Enhanced visual realism

---

## Evaluation Metrics

| Metric              | Purpose                               |
| ------------------- | ------------------------------------- |
| NIQE                | Natural Image Quality Evaluation      |
| BRISQUE             | No-Reference Image Quality Assessment |
| CLIP Similarity     | Semantic Consistency Measurement      |
| ΔE Color Difference | Color Adaptation Accuracy             |

---

## Challenges

* High GPU memory requirements for diffusion models
* Increased inference time for GPSDiffusion
* Large model storage requirements
* Occasional image downsampling for computational efficiency

Despite these challenges, the framework successfully produces realistic image composites across diverse scenarios.

---

## Future Work

* Multi-Object Scene Composition
* Video-Based Compositing
* Real-Time Shadow Generation
* Automatic Object Placement
* Improved Lighting Estimation
* Fully Automated End-to-End Pipeline

---

## Technologies Used

* Python
* OpenCV
* NumPy
* PyTorch
* Hugging Face Transformers
* Gradio
* U²Net
* MODNet
* ViTMatte
* BRIA RMBG
* GPSDiffusion



## Authors

### Group 19

* Basetti Sai Viswas (314561003)
* Amanuel Kerebo (314540002)
* Sree Nidhi (314540036)

---

## Course

Introduction to Visual Effects and Motion Graphics (IVMFX)

National Yang Ming Chiao Tung University (NYCU)

Spring 2026

