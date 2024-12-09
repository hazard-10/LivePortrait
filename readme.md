# Real-time Talking Head

This project provides comprehensive training and inference code for generating real-time talking head animations. By building upon state-of-the-art methodologies, this work seeks to advance the field of realistic portrait animation with a focus on real-time performance, natural expressiveness, and audio synchronization.

![Demo Video](https://github.com/hazard-10/LivePortrait/blob/main/366_1733367920.gif)

---

## Introduction

This project is rooted in the foundational contributions of **LivePortrait**, a framework for animating static portrait images based on motion retargeting from driving videos. LivePortrait introduces advanced techniques such as stitching and retargeting, enabling the seamless transfer of facial movements from a source video to a target static image. Its computational efficiency and visual fidelity make it an excellent foundation for applications demanding real-time portrait animation. 

In addition, **DiffPoseTalk** introduces an approach to speech-driven, stylistic 3D facial animation and head pose generation using diffusion models. The framework employs a style encoder that extracts stylistic features from reference videos, enabling the generation of diverse animations beyond the limitations of traditional one-hot encoding. 

While these methods offer an impressive baseline, its application in dynamic, real-time settings necessitates further refinements to enhance synchronization and interactivity. By leveraging the strengths of both LivePortrait and DiffPoseTalk, this project introduces a novel system for real-time talking head generation, blending high-quality animation with real-time responsiveness. 

---

## Contributions

This project makes contributions to the field of real-time portrait animation by improving upon the LivePortrait framework. The enhancements include:

### 1. **Enhanced Lip Synchronization**

This work introduces refined techniques and parameter adjustments to achieve precise alignment between lip movements and the spoken audio. By reducing mismatches and delays, the generated animations offer a more immersive and convincing experience, particularly in applications where accurate speech representation is essential.

### 2. **Improved Facial Expressiveness**

Facial expressions and movements, including gaze direction and head pose, are key elements of natural human communication. This project incorporates tuning of facial animation parameters to enhance the realism and expressiveness of the generated portraits. Specific improvements include:
- **Head Pose Dynamics:** Adjusting head movements to align with the context of speech and emotional expression, ensuring fluid and believable animations.
- **Subtle Expressions:** Incorporating nuanced changes in facial expressions, such as micro-expressions, to add depth and authenticity to the generated animations.
- **Gaze Contact:** Refining the gaze direction to control eye movement.

### 3. **Real-Time Rendering Capability**

This work is the optimization of the LivePortrait inference framework to enable real-time performance. By leveraging efficient computational techniques and system-level optimizations, this project ensures that high-quality animations can be rendered at interactive speeds. Key achievements include:
- Reducing the computational overhead of the retargeting process.
- Optimizing rendering pipelines to maintain low latency.
- Parallelizing with CUDA streams to improve performance compared to sequential execution. 

---

## Installation

To install and set up the system, follow these steps to ensure all required dependencies are satisfied. Detailed instructions are included in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

---

### Inference Pipeline

The inputs to this model consist of a portrait image with the target character and a driving audio. The entire inference pipeline can be illustrated as below. Full inference example [here](/inference/inference_final.ipynb).

<div align="center">
  <img src="/assets/docs/inference_pipeline.png" alt="Inference Pipeline" width="40%">
</div>

First, the portrait image and the input audio will pass through the LivePortrait pipeline and wave2vec pipeline to extract the facial feature embeddings and audio feature embeddings respectively. Then the character motion feature embeddings will be generated using the driving audio embeddings through the DiT model. Combining the facial features and the generated motion embeddings, the LivePortrait model can generate the frames with animated target character.

---

### Inference Analysis

Initial analysis shows that frame generation takes the majority of the inference time, compared to feature embedding generation and motion generation. This is because frame generation operates in pixel space, which involves significantly more computation. Additionally, the LivePortrait model is a larger model with approximately 100M parameters. While motion generation time depends on the context length, we choose to use 10 frames as context to generate 65 frames to minimize streaming latency (25 FPS).

<div align="center">
  <img src="/assets/docs/inference_performance.png" alt="Inference Performance" width="70%">
</div>

PyTorch profiler results on the LivePortrait model indicate that a significant amount of time is spent transferring tensors between the CPU and GPU, causing computation delays. To address this, pinned memory is utilized for faster CPU-GPU transfers, and CUDA streams are employed to overlap data transfer with computation. Additionally, by leveraging models with optimized compute graphs, the inference time of the LivePortrait model is reduced by 32.7%.

<div align="center">
  <img src="/assets/docs/profiler.png" alt="Profiler Results" width="70%">
</div>

<div align="center">
  <img src="/assets/docs/compare.png" alt="Comparison Analysis" width="70%">
</div>

---

### Gradio Interface

We also provide a [Gradio](https://github.com/gradio-app/gradio) interface for a better experience, just run by:
```
python inference/demo.py
```
and then access the demo page at `http://127.0.0.1:9995/` in browers.

![gradio](/assets/docs/gradio.gif)


---

## References

This project builds upon the methodologies and ideas presented in the following key papers. These works provide the foundational frameworks and inspiration for our advancements in real-time talking head generation:

1. **LivePortrait**  
   - **Title**: "LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control"  
   - **Authors**: Jianzhu Guo et al.  
   - **Description**: Introduced a highly efficient framework for animating static portrait images using motion retargeting techniques, emphasizing real-time performance and computational efficiency.  
   - **Link**: [LivePortrait Paper](https://arxiv.org/abs/2407.03168)

2. **DiffPoseTalk**  
   - **Title**: "DiffPoseTalk: Speech-Driven Stylistic 3D Facial Animation and Head Pose Generation via Diffusion Models"  
   - **Authors**: Zhiyao Sun et al.  
   - **Description**: Proposed a speech-driven framework for generating stylistic 3D facial animations and head pose dynamics using diffusion models. Emphasized diversity in animation through style embeddings extracted from reference videos.  
   - **Link**: [DiffPoseTalk Paper](https://arxiv.org/abs/2310.00434)


