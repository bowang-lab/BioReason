<h1 align="center">
🧬 BioReason<br>Incentivizing Multimodal Biological Reasoning<br>within a DNA-LLM Model
</h1>

<p align="center">
  <a href="https://www.arxiv.org/abs/2505.23579" target="_blank"><img src="https://img.shields.io/badge/arXiv-2505.23579-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://github.com/bowang-lab/BioReason"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://bowang-lab.github.io/BioReason/"><img src="https://img.shields.io/badge/Website-Online-00B89E?style=for-the-badge&logo=internet-explorer&logoColor=white" alt="Website"></a>
  <a href="https://huggingface.co/collections/wanglab/bioreason-683cd17172a037a31d208f70"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a>
</p>

<br>

## Updates [Jun 10, 2025]
- We are integrating vLLM to improve the speed and efficiency of the GRPO pipeline. We expect this to be pushed by end of week.
- Checkpoints along with the custom DNA-LLM model class will be released on HuggingFace by end of week.
- More training results with GRPO will be shared soon.

<br>

## Abstract

Unlocking deep, interpretable biological reasoning from complex genomic data is a major AI challenge hindering scientific discovery. Current DNA foundation models, despite strong sequence representation, struggle with multi-step reasoning and lack inherent transparent, biologically intuitive explanations. We introduce BioReason, a pioneering architecture that, for the first time, deeply integrates a DNA foundation model with a large language model (LLM). This novel connection enables the LLM to directly process and reason with genomic information as a fundamental input, fostering a new form of multimodal biological understanding. BioReason's sophisticated multi-step reasoning is developed through supervised fine-tuning and targeted reinforcement learning, guiding the system to generate logical, biologically coherent deductions. On biological reasoning benchmarks including KEGG-based disease pathway prediction—where accuracy improves from 88% to 97%—and variant effect prediction, BioReason demonstrates an average 15% performance gain over strong single-modality baselines.

<br>

## Key Contributions

• **Novel multimodal architecture**: The first successful integration of a DNA foundation model with an LLM, establishing a new methodology for AI-driven biological studies.

• **Advanced reasoning methodology**: A systematic training approach combining supervised fine-tuning and reinforcement learning that incentivizes multi-step biological reasoning.

• **New biological reasoning benchmarks**: Development and curation of novel benchmarks for evaluating biological reasoning capabilities, including an annotated reasoning dataset for gene pathway and disease prediction from KEGG.

• **Empirical performance improvements**: Demonstration that BioReason outperforms both DNA foundation models and LLMs used independently or in simple combination, with average performance gains of 15%+ over baseline.

• **Interpretable reasoning traces**: A mechanism for generating step-by-step biological reasoning traces that provide interpretable predictions, enhancing scientific insight and hypothesis generation.

<br>

## Datasets

The datasets used to train and evaluate BioReason can be found on our [HuggingFace collection](https://huggingface.co/collections/wanglab/bioreason-683cd17172a037a31d208f70) with detailed download and usage instructions.

<br>

## Checkpoints

We will release the checkpoints soon!

<br>

## Installation

### Prerequisites
- Python 3.11+
- CUDA/GPU for best performance

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/bowang-lab/BioReason.git
cd BioReason

# Install package
pip install -e .
```

<br>

## Results

### KEGG-Derived Biological Reasoning Task
Performance comparison on 290 test datapoints for multi-step mechanistic reasoning:

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|---------|
| [DNA] NT - 500M | 86.55 | 69.76 | 73.23 | 66.61 |
| [DNA] Evo2 - 1B | 88.28 | 72.43 | 75.23 | 69.83 |
| [LLM] Qwen3 - 1B | 85.17 | 65.71 | 71.39 | 64.19 |
| [LLM] Qwen3 - 4B | 93.48 | 85.44 | 88.31 | 86.72 |
| [DNA-LLM] NT + Qwen3 - 1B | 88.42 | 72.13 | 75.42 | 71.91 |
| [DNA-LLM] NT + Qwen3 - 1B (+RL) | 89.66 | 74.11 | 78.82 | 72.96 |
| [DNA-LLM] NT + Qwen3 - 4B | 96.90 | **89.03** | **90.99** | **89.38** |
| [DNA-LLM] Evo2 + Qwen3 - 1B | 90.42 | 75.62 | 77.42 | 73.91 |
| [DNA-LLM] Evo2 + Qwen3 - 4B | **97.24** | 86.30 | 86.75 | 87.25 |

### Variant Effect Prediction Benchmarks
Performance on pathogenic/benign classification:

| Model | Variant Effect - Coding | | Variant Effect - Non-SNV | |
|-------|------------|----------|------------|----------|
| | Accuracy | F1-Score | Accuracy | F1-Score |
| [DNA] NT - 500M | 60.91 | 45.20 | 67.93 | 65.97 |
| [DNA] Evo2 - 1B | 70.07 | 49.19 | 76.17 | 66.51 |
| [LLM] Qwen3 - 1B | 46.55 | 34.82 | 70.67 | 76.21 |
| [LLM] Qwen3 - 4B | 48.99 | 39.58 | 61.86 | 67.60 |
| [DNA-LLM] NT + Qwen3 - 1B | 55.58 | 54.50 | 72.82 | 76.93 |
| [DNA-LLM] NT + Qwen3 - 4B | 60.94 | 55.66 | 65.59 | 73.00 |
| [DNA-LLM] Evo2 + Qwen3 - 1B | 72.83 | 68.90 | **88.20** | **89.91** |
| [DNA-LLM] Evo2 + Qwen3 - 4B | **80.21** | **80.00** | 83.85 | 85.02 |

<br>

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{fallahpour2025bioreasonincentivizingmultimodalbiological,
      title={BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model}, 
      author={Adibvafa Fallahpour and Andrew Magnuson and Purav Gupta and Shihao Ma and Jack Naimer and Arnav Shah and Haonan Duan and Omar Ibrahim and Hani Goodarzi and Chris J. Maddison and Bo Wang},
      year={2025},
      eprint={2505.23579},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.23579}, 
}
```

<br>

## Authors

- **Adibvafa Fallahpour**¹²³⁵ * (adibvafa.fallahpour@mail.utoronto.ca)
- **Andrew Magnuson**¹² *
- **Purav Gupta**¹² *
- **Shihao Ma**¹²³
- **Jack Naimer**¹²³
- **Arnav Shah**¹²³
- **Haonan Duan**¹²
- **Omar Ibrahim**³
- **Hani Goodarzi**†⁴⁶
- **Chris J. Maddison**†¹²⁷
- **Bo Wang**†¹²³

¹ University of Toronto ² Vector Institute ³ University Health Network (UHN) <br>
⁴ Arc Institute ⁵ Cohere ⁶ University of California, San Francisco ⁷ Google DeepMind

<br>
* Equal contribution <br>
† Equal advising

---

<p align="center">
Made with ❤️ at University of Toronto, Vector Institute, and University Health Network
</p>
