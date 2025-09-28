# PT²-LLM: Post-Training Ternarization for Large Language Models

Xianglong Yan, Chengzhu Bao, [Zhiteng Li](https://zhitengli.github.io), Tianao Zhang, [Haotong Qin](https://htqin.github.io/), [Ruobing Xie](https://ruobingxie.github.io/), Xingwu Sun, and [Yulun Zhang](http://yulunzhang.com/), "PT²-LLM: Post-Training Ternarization for Large Language Models", arXiv, 2025

#### 🔥🔥🔥 News

- **2025-9-27:** This repo is released.

---

> **Abstract:** Large Language Models (LLMs) have shown impressive capabilities across diverse tasks, but their large memory and compute demands hinder deployment. Ternarization has gained attention as a promising compression technique, delivering substantial size reduction and high computational efficiency. However, its potential in the post-training quantization (PTQ) setting remains underexplored, due to the challenge of training-free parameter optimization and the quantization difficulty posed by outliers and dispersed weights. To address these issues, we propose PT$^2$-LLM, a post-training ternarization framework tailored for LLMs. At its core is an Asymmetric Ternary Quantizer equipped with a two-stage refinement pipeline: (1) Iterative Ternary Fitting (ITF), which alternates between optimal ternary grid construction and flexible rounding to minimize quantization error, and (2) Activation-aware Grid Alignment (AGA), which further refines the ternary grid to better match full-precision outputs. In addition, we propose a plug-and-play Structural Similarity-based Reordering (SSR) strategy that leverages inter-column structural similarity to ease quantization and mitigate outlier effects, further enhancing overall performance. Extensive experiments demonstrate that PT$^2$-LLM delivers competitive performance against state-of-the-art (SOTA) 2-bit PTQ methods with lower memory cost, while also accelerating both prefill and decoding to achieve end-to-end speedup. The code and models will be available at https://github.com/XIANGLONGYAN/PT2-LLM.

![](figs/overview.png)

---
