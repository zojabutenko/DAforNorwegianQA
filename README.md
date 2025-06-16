# Data Augmentation for Question Answering in Norwegian

This repository contains the code, data, and trained models developed for the master's thesis "Data Augmentation for Question Answering in Norwegian", submitted at the University of Oslo in Spring 2025.

**Author**: Zoia Butenko

**Supervisors**: Vladislav Mikhailov, Sondre Wold, Lilja Øvrelid

---

## Abstract

> The advancement of the NLP field has accelerated in recent years, mainly due to the emergence of transformer-based architectures and Large Language Models (LLMs). In the field of Question Answering (QA), generating synthetic question-answer pairs has become a well-established data augmentation approach to improve QA models. However, QA data augmentation remains understudied in the context of less resourced languages, such as Norwegian.  
>  
> In this thesis, we analyze three data augmentation approaches for the task of Norwegian extractive QA. We compare different configurations of the DA approaches based on generative large language models, machine translation, and back-translation and discuss how the size and domain of the augmented data affect the performance of three monolingual and multilingual encoder-based QA models.  
>  
> Our key findings indicate that the multilingual model greatly benefits from data augmentation, with F1 improving on average 3% relative to the baseline. We empirically show that combining the synthetic data with as little as 25% human-labeled data significantly improves the model performance. We also find that the size of the augmented data plays a crucial role in the performance, and its effect varies across data augmentation approaches. In addition, we evaluate the quality of synthetic data with traditional text generation metrics and find that they are reflective of downstream performance. Finally, we discuss the implications of our findings on other low-resource languages, especially other languages and variants of Norwegian. We make all our augmentation datasets and best fine-tuned QA models publicly available.

---

## Repository Structure

...
