---
slayout: post
title: A Continual Learning Framework for Uncertainty-Aware Interactive Segmentation
cover-img: /assets/img/CT_image.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/CT_image.jpg
subtitle: Continual learning and interactive segmentation
tags: [Deep Learning, Interactive Segmentation, Continual Learning]
comments: true
---

**본 논문은 Rochester Institute of Technology 에서 2021 AAAI 에 발표한 논문입니다.**

## Abstract



---



## Introduction

---



## Related Works

---



## The Framework of Continual Learning for Interactive Segmentation

---

#### Overview of the Architecture

### Task Formulation for Interactive Segmentation

### Continual Knowledge Learning through Bayesian Nonparametric Modeling

### Variational Inference

### Predicting Initial Segmentation

### Interactive Segmentation

#### Uncertainty Estimation

#### Propagating User Annotations

#### Learning from User Annotations



## Experiments

---

#### Experimental setup

#### Comparison baselines and evaluation metrics

#### Performance comparison

#### Interpretability

---



## Conclusion

1. 본 논문에서는 Interactive segmentation task 를 continual-learning problem 으로 해결하려고 하였고, segmentation problem 에서 이전 task 에서 발생하는 catastrophic forgetting 을 방지해주는 framework 을 제안하였다.
2. Uncertainty information 을 활용하여 user 들에게 informative guidance 를 해줌으로써 segmentation annotation 을 위한 최소한의 노력을 들일 수 있도록 할 수 있다.
3. 현재 interaction 이 click 으로 한정되어 있는데 scribble, boxes 등으로 확장하여 향후 연구 방향을 설정할 수 있다.
4. (논문이 쓰여질 당시 SOTA) ImageNet 에서 ResNet-50 을 기반으로 한 linear evaluation protocol 에서 SOTA 성능을 보여주었으며, ResNet-200 을 사용했을 때 기존 sota 모델 대비 30% 적은 수의 parameter 를 사용하고도 top-1 accuracy 79.6% 의 성능을 보여줬습니다. (기존 성능 76.8%)
5. 그럼에도 불구하고, BYOL 은 vision task 에서만 사용될 수 있는 augmentation 방식을 채택하고 있습니다. BYOL 이 일반적으로 사용될 수 있으려면 다른 modalities 들 (audio, video, text etc.) 에서도 통용되는 augmentation 방식들도 사용해볼 필요가 있습니다. 하지만 적절한 augmentation 방식들을 searching 하는 것은 많은 노력이 필요하기 때문에 이를 자동화 하여 augmentation 을 searching 하는 방식을 찾는 것이 중요하다고 말하고 있습니다.

