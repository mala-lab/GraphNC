<div align="center">
<img src="GraphNC logo.png" width="260">


  <h2><b> Normality Calibration in Semi-supervised Graph Anomaly Detection </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/mala-lab/GraphNC?color=green)
![](https://img.shields.io/github/stars/mala-lab/GraphNC?color=yellow)
![](https://img.shields.io/github/forks/mala-lab/GraphNC?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)
[![arXiv](https://img.shields.io/badge/arXiv-2510.02014-b31b1b)](https://arxiv.org/abs/2510.02014)
</div>


# Overview
We propose **GraphNC**, a <u>graph</u> <u>n</u>ormality <u>c</u>alibration framework that leverages both labeled and unlabeled data to calibrate the normality from a teacher, namely a pre-trained semi-supervised GAD model, jointly in **anomaly score** and **representation** spaces.

GraphNC includes two main components: anomaly <u>score</u> <u>d</u>istribution <u>a</u>lignment (**ScoreDA**) and perturbation-based <u>norm</u>ality <u>reg</u>ularization (**NormReg**). **ScoreDA** optimizes the anomaly scores of our model by aligning them with the score distribution yielded by the teacher. Since the teacher provides accurate scores for most normal nodes and a subset of anomaly nodes, this alignment effectively pulls the anomaly scores of the two classes toward opposite ends, resulting in more separable anomaly scores.

To mitigate the misleading effects of inaccurate teacher scores, **NormReg** is designed to regularize normality in the representation space. Specifically, it encourages more compact representations of normal nodes by minimizing a perturbation-guided consistency loss solely on labeled nodes.




<div align="center"><img src="pipeline.png" width="92%"></div>

## Key Highlights

- We propose **VerifyMAS**, an error-first hypothesis verification framework for failure attribution in LLM-MAS. It decomposes failure attribution into error validation and fine-grained faulty agent localization, providing a principled solution for agentic failure attribution of both global and local errors.  

- We propose a fine-tuning strategy tailored to the hypothesis verification approach, in which trajectory-level verification samples and agent-localization supervision are collected and leveraged to fine-tune an LLM verifier model under the VerifyMAS framework. This substantially enhances our model in failure diagnosis of in-distribution trajectories while preserving robust generalization to out-of-distribution trajectories. The dataset will be released to promote more advances in this line.  

- Extensive experiments on *Aegis-Bench* and *Who&When* demonstrate that VerifyMAS consistently improves diverse open-source and proprietary models. We further validate its effectiveness under the SFT setting, where hypothesis-verification-based fine-tuning strengthens in-distribution diagnostic ability while preserving out-of-distribution generalization.




## Main Results

<div align="center"><img src="main_table.png" width="98%"></div>



The code and model will be uploaded soon ! 
