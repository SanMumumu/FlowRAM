
# \$FlowRAM\$: Grounding Flow Matching Policy with Region-Aware Mamba Framework for Robotic Manipulation

<h4 align="center">Sen Wang<sup>1*</sup>, Le Wang<sup>1†</sup>, Sanping Zhou<sup>1</sup>, Jingyi Tian<sup>1</sup>, Jiayi Li<sup>1</sup>, Haowen Sun<sup>1</sup>, Wei Tang<sup>2</sup></h4>
<h4 align="center"><sup>1</sup>National Key Laboratory of Human-Machine Hybrid Augmented Intelligence, Xi’an Jiaotong University</h4>
<h4 align="center"><sup>2</sup>University of Illinois at Chicago</h4>
<h4 align="center">
  <a href="https://sanmumumu.github.io/FlowRAM/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/pdf/2506.16201"><strong>ArXiv</strong></a>
  |
  <a href="https://blog.csdn.net/weixin_45751396/article/details/149784481?spm=1001.2014.3001.5502"><strong>Blog (In Chinese)</strong></a>
</h4>
<div align="center">
  <img src="asserts\framework.png" alt="FlowRAM Framework" width="90%">
</div>



## Abstract
Robotic manipulation in high-precision tasks is essential for numerous industrial and real-world applications where accuracy and speed are required. Yet current diffusion-based policy learning methods generally suffer from low computational efficiency due to the iterative denoising process during inference. Moreover, these methods do not fully explore the potential of generative models for enhancing information exploration in 3D environments. In response, we propose FlowRAM, a novel framework that leverages generative models to achieve region-aware perception, enabling efficient multimodal information processing. Specifically, we devise a Dynamic Radius Schedule, which allows adaptive perception, facilitating transitions from global scene comprehension to fine-grained geometric details. Furthermore, we integrate state space models to integrate multimodal information, while preserving linear computational complexity. In addition, we employ conditional flow matching to learn action poses by regressing deterministic vector fields, simplifying the learning process while maintaining performance. We verify the effectiveness of the FlowRAM in the RLBench, an established manipulation benchmark, and achieve state-of-the-art performance. The results demonstrate that FlowRAM achieves a remarkable improvement, particularly in high-precision tasks, where it outperforms previous methods by 12.0\% in average success rate. Additionally, FlowRAM is able to generate physically plausible actions for a variety of real-world tasks in less than 4 time steps, significantly increasing inference speed.




## 💻 Installation

See [install.md](insatll.md) for installation instructions.

## 📚 Data

FlowRAM leverages the [RLBench](https://github.com/stepjam/RLBench) framework to generate expert demonstrations, including *precision-focused tasks* for high-accuracy manipulation. Generated data is saved in:

```bash
$YOUR_REPO_PATH/FlowRAM/data/
```

We follow RLBench’s data generation pipeline for consistency and scalability.



## 🛠️ Usage

Scripts for training and evaluation are included in the `scripts/` & `online_evaluation_rlbench/` directory.

1. **Train FlowRAM in GNFactor setup**:

   ```bash
   bash scripts/gnfactor_train.sh
   ```
2. **Train FlowRAM in Precise setup**:

   ```bash
   bash scripts/precise_train.sh
   ```
2. **Evaluate a policy**:

   ```bash
   bash online_evaluation_rlbench\eval_peract.sh
   ```



## 🤖 Real-world Deployments

FlowRAM supports deployment on a **6-DoF UR5 arm with Robotiq gripper**, achieving robust manipulation across six real-world tasks.

<div align="center">
  <img src="asserts\real.jpg" alt="FlowRAM Framework" width="50%">
</div>




## 🚧 TODO
- [ ] 📝 Formatting code for release
- [ ] 📦 Open-sourcing pretrained weights
- [ ] ⏳ Currently working on other projects, will release when time permits.  

















## 🏷️ License

This repository is licensed under the MIT License.


## 🙏 Acknowledgements

Our work builds on **3D Diffuser Actor**, **PointMamba**, and **Mamba**. We thank these projects for their inspiring contributions.


## 👍 Citation

```
@inproceedings{wang2025flowram,
  title={FlowRAM: Grounding Flow Matching Policy with Region-Aware Mamba Framework for Robotic Manipulation},
  author={Wang, Sen and Wang, Le and Zhou, Sanping and Tian, Jingyi and Li, Jiayi and Sun, Haowen and Tang, Wei},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={12176--12186},
  year={2025}
}
```

