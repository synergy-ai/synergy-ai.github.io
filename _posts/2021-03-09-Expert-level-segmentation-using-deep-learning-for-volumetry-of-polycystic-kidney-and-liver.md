---
layout: post
title: Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++
cover-img: /assets/img/CT_image.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/CT_image.jpg
subtitle: Efficient Interactive Labelling
tags: [Segmentation, Interactive, Annotation, CVPR, Deep Learning]
comments: true
---

## Abstract

이 논문은 **CVPR 2018 에 소개된 논문**으로서, 2017 년 같은 학회에 소개된 polygon-rnn 을 (Human-intervened 반자동 레이블링 모델) 더 효율적이고 높은 성능을 개선한 **Polygon-RNN ++** 를 소개하고 있습니다.

본 논문이 기존 모델에 비해 향상시킨 **3가지 포인트**는 다음과 같습니다.

1. **새로운 CNN 인코더 구조**를 소개했으며
2. **Reinforcement Learning (강화학습)**을 활용하여 모델을 효과적으로 학습하였고
3. **Graph Neural Network (GNN) 을 활용**하여 **output 의 해상도를 증가**시켰습니다.

기존 모델에 비해 **10%, 16% 의 absolute, relative mean IoU 향상**이 있었으며 annotator 가 기존 모델에 비해 **50% 의 더 적은 클릭**을 통해 annotation 할 수 있습니다.

또한 **well-generalized** 모델이기 때문에 **cross-domain task** 에서 좋은 성능을 보이며 **새로운 데이터셋에서도 더 높은 성능**을 낼 수 있도록 적용할 수 있습니다.

## Introduction



## Materials and Methods



## Results



## Discussions



## Conclusion

1. 본 논문에서는 **Object instance segmentation** 을 위한 **Polygon-RNN++** 를 제안하였고, 이를 활용하여 segmentation dataset 에서 interactive annotation 을 잘 수행할 수 있음을 보였습니다.
2. 기존의 **Polygon-RNN** 을 향상시켜 모델을 구성했는데, 결과적으로 **automatic, interactive** 두 가지 모드에서 모두 더 좋은 성능을 보입니다.
3. **Noisy annotator** 들에게도 **robust** 하게 작동함을 보였으며 **새로운 도메인에서도 잘 작동**하는 것을 보였습니다. 더 나아가, **Online fine-tuning scheme 을 활용**하여 새로운 **out-of-domain dataset** 에서도 효과적으로 쓰일 수 있음을 보였습니다.