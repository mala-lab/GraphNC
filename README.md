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




<div align="center"><img src="framework.png" width="92%"></div>

## Key Highlights

- We introduce **GraphNC**, a novel **graph normality calibration framework** that leverages both **labeled normal nodes** and **unlabeled data** to calibrate normality learning from the teacher model in both the **score space** and **feature space** for **semi-supervised GAD**.

- In **GraphNC**, we introduce two new components: **anomaly score distribution alignment (ScoreDA)** and **perturbation-based normality regularization (NormReg)**. **ScoreDA** aligns the scores of our model with those of the pre-trained teacher model, which helps push the anomaly scores of the two classes toward **opposite ends**. Meanwhile, **NormReg** mitigates the impact of inaccurate teacher scores by enforcing **representation compactness among normal nodes** during alignment.

- **GraphNC** is a **flexible plug-and-play framework** where different teacher models can be incorporated, achieving **consistently enhanced GAD performance** across three types of teacher models, including **data reconstruction-based**, **one-class-based**, and **anomaly-generation-based** approaches.



## Main Results

<div align="center"><img src="main_table.png" width="98%"></div>



The code and model will be uploaded soon ! 
