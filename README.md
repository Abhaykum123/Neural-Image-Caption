
# Neural Image Caption Generation with Visual Attention

This repository implements an image‑to‑text caption generation model using visual attention. The goal is to automatically generate natural language descriptions for images by allowing the model to *attend* to meaningful regions of the image during caption generation.

---

## Table of Contents

1. [Problem Statement](#problem-statement)  
2. [Motivation & Background](#motivation--background)  
3. [Related Work](#related-work)  
4. [Model Architecture](#model-architecture)  
5. [Implementation Details](#implementation-details)  
6. [Datasets](#datasets)  
7. [Training Procedure](#training-procedure)  
8. [Evaluation & Metrics](#evaluation--metrics)  
9. [Results](#results)  
10. [Usage](#usage)  
11. [Requirements](#requirements)  
12. [Contributing](#contributing)  
13. [License](#license)  
14. [References](#references)

---

## Problem Statement

Given an input image, generate a descriptive caption in natural language that captures the objects, actions, and relevant relationships in the image.

Key challenges:

- Understanding object presence, attributes and spatial relationships.  
- Generating coherent sentences with proper grammar and composition.  
- Focusing on relevant image regions (via attention) rather than summarizing globally only.  

---

## Motivation & Background

- Traditional image captioning models encode the whole image into a fixed vector (using a CNN) and then use an RNN decoder to generate text. This loses spatial information, especially when multiple objects or fine spatial detail matter.  
- The *attention mechanism* allows the model to dynamically focus on different parts of the image at each decoding step, improving descriptive richness and interpretability.  

---

## Related Work

Here are some of the major foundational works and more recent advances in this area:

| Paper | Key Contribution |
|---|---|
| *“Show, Attend and Tell: Neural Image Caption Generation with Visual Attention”* by Xu et al., 2015 | Introduced both **soft** (differentiistic) and **hard** (stochastic) attention mechanisms in the image captioning setting. Demonstrated how attention weights can reveal which regions the model focuses on when generating each word. ([arxiv.org](https://arxiv.org/abs/1502.03044)) |
| *SCA‑CNN: Spatial and Channel‑wise Attention in Convolutional Networks for Image Captioning* (Chen et al., 2016) | Proposed attention both over spatial locations and the feature channels to better decide *where* and *what* to attend to in images. ([arxiv.org](https://arxiv.org/abs/1611.05594)) |
| *Knowing When to Look: Adaptive Attention via a Visual Sentinel* (Lu et al., 2016) | Not all words in the caption require attending to the image (e.g. “the”, “of”). This model learns when to attend to the image and when to rely more heavily on the language model via a *visual sentinel*. ([arxiv.org](https://arxiv.org/abs/1612.01887)) |
| *Adaptively Aligned Image Captioning via Adaptive Attention Time* (Huang et al., 2019) | Provides flexibility in the alignment by learning how many attention steps to take before generating each word. ([arxiv.org](https://arxiv.org/abs/1909.09060)) |

Your project builds upon these ideas (especially the Show, Attend and Tell framework), adopting an encoder‑decoder architecture with visual attention.

---

## Model Architecture

1. **Encoder (CNN)**  
   - A pre‑trained convolutional neural network (e.g. VGG, ResNet, Inception) is used to extract visual features from the image.  
   - Features are usually taken from a convolutional layer (not the fully connected layer) so that spatial layout is preserved (e.g. feature map of size *H × W × D*).  

2. **Attention Mechanism**  
   - At each time step in decoding, attention weights are computed over spatial regions of the CNN feature map.  
   - Soft (differentiable) attention is typically used (weighted sum of features), although hard (stochastic) attention is possible.  
   - Incorporate mechanisms like channel attention or visual sentinel, if applicable.  

3. **Decoder (RNN / LSTM / GRU)**  
   - Takes the attended features plus the previous word (or embedding) and hidden state to predict the next word.  
   - Word embeddings layer → RNN unit → fully connected layer → softmax over vocabulary.  

4. **Optional Enhancements**  
   - Beam search at inference time to improve caption generation.  
   - Regularization (dropout, weight decay).  
   - Possibly scheduled sampling or teacher forcing, depending on how you train.  

---

## Implementation Details

- **Programming environment**: Jupyter notebook / Python scripts  
- **Libraries used**: e.g. PyTorch / TensorFlow, NumPy, PIL / OpenCV, etc.  
- **Tokenizer**: How the vocabulary is built, max sentence length, handling of unknown words, etc.  
- **Attention implementation**: soft/hard, visualization method, how attention maps are extracted.  
- **Checkpoints**: Explanation of saved encoder/decoder weights, tokenizer, etc.  

---

## Datasets

- MS COCO (Common Objects in Context)  
- Flickr8k, Flickr30k (if used)  
- Any preprocessing you did (resizing, cropping, normalization, train‑val‑test split)  

---

## Training Procedure

- Hyperparameters: learning rate, batch size, number of epochs  
- Optimizer: e.g. Adam, RMSProp, etc.  
- Loss function: cross‑entropy over words, possibly with additional regularization / attention penalties.  
- Teacher forcing: whether used (i.e., feeding ground‑truth previous words vs model’s own predictions during training).  
- Any tricks: gradient clipping, learning rate decay, early stopping.  

---

## Evaluation & Metrics

Common metrics for image captioning include:

- BLEU (1 to 4)  
- METEOR  
- ROUGE‑L  
- CIDEr  
- SPICE (if applicable)  

Also qualitative evaluation: show sample images + captions + attention maps to see where the model is focusing.  

---

## Results

_(Insert your experimental results here)_

- Quantitative scores on chosen dataset(s)  
- Sample generated captions vs ground truth captions  
- Visualizations of attention maps  

---

## Usage

How to run your code / how to generate captions:

```bash
# Example commands

# To train the model
python train.py --data_dir path/to/images --annotations path/to/annotations --epochs XX

# To generate captions on test images
python generate.py --image_dir path/to/test_images --checkpoint path/to/model.ckpt
```

---

## Requirements

- Python version  
- Libraries and versions (TensorFlow or PyTorch, etc.)  
- GPU usage (if required)  
- Memory / disk requirements  

---

## Contributing

If you want others to contribute, you can include:

- Coding style guidelines  
- How to report issues / feature requests  
- Branching model / pull request process  

---

## License

Include your license (MIT, Apache, etc.). The code in this repo is under **MIT License**.

---

## References

1. Xu, Kelvin; et al. *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*. 2015. [arxiv.org](https://arxiv.org/abs/1502.03044)  
2. Chen, Long; et al. *SCA‑CNN: Spatial and Channel‑wise Attention in Convolutional Networks for Image Captioning.* 2016. [arxiv.org](https://arxiv.org/abs/1611.05594)  
3. Lu, Jiasen; et al. *Knowing When to Look: Adaptive Attention via a Visual Sentinel for Image Captioning.* 2016. [arxiv.org](https://arxiv.org/abs/1612.01887)  
4. Huang, Lun; et al. *Adaptively Aligned Image Captioning via Adaptive Attention Time.* 2019. [arxiv.org](https://arxiv.org/abs/1909.09060)  
5. Sasibhooshan, Santhoshkumar; et al. *Image caption generation using Visual Attention Prediction and Contextual Spatial Relation Extraction.* Journal of Big Data, 2023. [journalofbigdata.springeropen.com](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00693-9)
