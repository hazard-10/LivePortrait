# Real-time talking head

This project provides training and inference code for a real-time talking head generation.

[Demo video](https://drive.google.com/file/d/14Wkd0aD27ho2SerSEQtjrxrrvtdOovhi/view?usp=sharing)

### Install

see requirements.txt

### 3rd party code used

The facial representation is from LivePortrait. For more details about LivePortrait, see the [LivePortrait Readme](./live_readme.md).

### Inference Pipeline

The inputs to this model consist of a portrait image with the target character and a driving audio. The entire inference pipeline can be illustrated as below. Full inference example [here](/inference/inference_final.ipynb).

![inference pipeline](/assets/docs/inference_pipeline.png)

First, the portrait image and the input audio will pass through the LivePortrait pipeline and wave2vec pipeline to extract the facial feature embeddings and audio feature embeddings respectively. Then the character motion feature embeddings will be generated using the driving audio embeddings through the DiT model. Combining the facial features and the generated motion embeddings, the LivePortrait model can generate the frames with animated target character.

### Inference Analysis

Initial analysis shows that frames generation takes the majority of the inference time, compared to feature embedding generation and motion generation. This is because frames generation needs to work in pixel space which involves significantly more computation and also LivePortriat model is a bigger model with around 100M parameters. While motion generation time depends on the context length, we choose to use 10 frames as context to generate 65 frames for shorter streaming latency (25 FPS).

![inference performace](/assets/docs/inference_performance.png)

PyTorch porfiler results on LivePortrait model shows that a lot of the time is spend on transferring tensors between CPU and GPU, causing computation delays. Pinned memory is therefore used for fast CPU-GPU transfers and cuda streams to overlap transfer with computation. Also, leveraging models with optimized compute graphs, inference time of LivePortrait model is reduced by 32.7%.

![profiler](/assets/docs/profiler.png)
<br>
![analysis](/assets/docs/compare.png)

### Gradio Interface

We also provide a [Gradio](https://github.com/gradio-app/gradio) interface for a better experience, just run by:
```
python inference/demo.py
```
and then access the demo page at `http://127.0.0.1:9995/` in browers.

![gradio](/assets/docs/gradio.gif)
