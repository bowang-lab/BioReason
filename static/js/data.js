const paper = {
  metadata: {
    title: "BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model",
    description: "A pioneering architecture that deeply integrates a DNA foundation model with a large language model (LLM) to enable multimodal biological understanding and reasoning, achieving significant performance improvements on biological reasoning benchmarks.",
    keywords: ["DNA Foundation Models", "Large Language Models", "Biological Reasoning", "Multimodal AI", "Genomics", "Machine Learning", "Bioinformatics"],
  },
  navbar: {
    home_link: "https://bowang-lab.github.io",
    more_research: [
      {
        name: "Bo Wang Lab",
        link: "https://bowang-lab.github.io",
      },
      {
        name: "Vector Institute",
        link: "https://vectorinstitute.ai",
      },
      {
        name: "University of Toronto",
        link: "https://www.utoronto.ca",
      },
      {
        name: "ArXiv",
        link: "https://arxiv.org/abs/2505.23579",
      },
    ],
  },
  authors: [
    {
      name: "Adibvafa Fallahpour",
      superscript: "1,2,3,5*",
      website: "https://www.linkedin.com/in/adibvafa-fallahpour/",
    },
    {
      name: "Andrew Magnuson",
      superscript: "1,2*",
      website: "mailto:andrew.magnuson@mail.utoronto.ca",
    },
    {
      name: "Purav Gupta",
      superscript: "1,2*",
      website: "mailto:purav.gupta@mail.utoronto.ca",
    },
    {
      name: "Shihao Ma",
      superscript: "1,2,3",
      website: "mailto:shihao.ma@mail.utoronto.ca",
    },
    {
      name: "Jack Naimer",
      superscript: "1,2,3",
      website: "mailto:jack.naimer@mail.utoronto.ca",
    },
    {
      name: "Arnav Shah",
      superscript: "1,2,3",
      website: "mailto:arnav.shah@mail.utoronto.ca",
    },
    {
      name: "Haonan Duan",
      superscript: "1,2",
      website: "mailto:haonan.duan@mail.utoronto.ca",
    },
    {
      name: "Omar Ibrahim",
      superscript: "3",
      website: "mailto:omar.ibrahim2@uhn.ca",
    },
    {
      name: "Hani Goodarzi",
      superscript: "4,6†",
      website: "mailto:hani.goodarzi@ucsf.edu",
    },
    {
      name: "Chris J. Maddison",
      superscript: "1,2,7†",
      website: "mailto:cmaddis@cs.toronto.edu",
    },
    {
      name: "Bo Wang",
      superscript: "1,2,3†",
      website: "mailto:bowang@vectorinstitute.ai",
    },
  ],
  affiliations: [
    {
      number: "1",
      name: "University of Toronto",
      logo: "static/images/uoft.png",
    },
    {
      number: "2", 
      name: "Vector Institute",
      logo: "static/images/vector.png",
    },
    {
      number: "3",
      name: "University Health Network",
      logo: "static/images/uhn.png",
    },
    {
      number: "4",
      name: "Arc Institute",
      logo: "static/images/arc.png",
    },
    {
      number: "5",
      name: "Cohere",
      logo: "static/images/cohere.png",
    },
    {
      number: "6",
      name: "University of California, San Francisco",
      logo: "static/images/ucsf.png",
    },
    {
      number: "7",
      name: "Google DeepMind",
      logo: "static/images/deepmind.png",
    },
  ],
  author_notes: {
    equal_contribution: "Equal Contribution",
    equal_advising: "Equal Advising",
  },
  link_items: [
    {
      name: "Paper",
      link: "https://arxiv.org/abs/2505.23579",
      icon: "ai ai-arxiv",
    },
    {
      name: "Code",
      link: "https://github.com/bowang-lab/BioReason",
      icon: "fab fa-github",
    },
    {
      name: "Dataset",
      link: "https://huggingface.co/collections/wanglab/bioreason-683cd17172a037a31d208f70",
      icon: "🤗",
    },
  ],
  content: {
    abstract:
      "Unlocking deep, interpretable biological reasoning from complex genomic data is a major AI challenge hindering scientific discovery. Current DNA foundation models, despite strong sequence representation, struggle with multi-step reasoning and lack inherent transparent, biologically intuitive explanations. We introduce BioReason, a pioneering architecture that, for the first time, deeply integrates a DNA foundation model with a large language model (LLM). This novel connection enables the LLM to directly process and reason with genomic information as a fundamental input, fostering a new form of multimodal biological understanding. BioReason's sophisticated multi-step reasoning is developed through supervised fine-tuning and targeted reinforcement learning, guiding the system to generate logical, biologically coherent deductions. On biological reasoning benchmarks including KEGG-based disease pathway prediction—where accuracy improves from 88% to 97%—and variant effect prediction, BioReason demonstrates an average 15% performance gain over strong single-modality baselines. BioReason reasons over unseen biological entities and articulates decision-making through interpretable, step-by-step biological traces, offering a transformative approach for AI in biology that enables deeper mechanistic insights and accelerates testable hypothesis generation from genomic data.",
    contributions: [
      "<strong>Novel multimodal architecture:</strong> The first successful integration of a DNA foundation model with an LLM, establishing a new methodology for AI-driven biological studies.",
      "<strong>Advanced reasoning methodology:</strong> A systematic training approach combining supervised fine-tuning and reinforcement learning that incentivizes multi-step biological reasoning.",
      "<strong>New biological reasoning benchmarks:</strong> Development and curation of novel benchmarks for evaluating biological reasoning capabilities, including an annotated reasoning dataset for gene pathway and disease prediction dataset from KEGG.",
      "<strong>Empirical performance improvements:</strong> Demonstration that BioReason outperforms both DNA foundation models and LLMs used independently or in simple combination, with average performance gains of 15%+ over baseline.",
      "<strong>Interpretable reasoning traces:</strong> A mechanism for generating step-by-step biological reasoning traces that provide interpretable predictions, enhancing scientific insight and hypothesis generation."
    ],
    sections: {
      architecture: {
        image: "./static/images/architecture.png",
        text: "BioReason operates on two primary input streams: (i) one or more genomic sequences, and (ii) textual queries. The DNA foundation model transforms each input DNA sequence into contextualized embeddings, while the Large Language Model serves as the primary reasoning engine and text generator. Key to this integration is the preparation of the DNA embedding block, where genomic information is integrated into the LLM's input by stacking DNA embeddings with embeddings of the user's query and special tokens such as <dna_start> and <dna_end>.",
      },
      experiments: {
        image: "./static/images/results.png",
        text: "BioReason's performance is evaluated on three datasets: KEGG-Derived Biological Reasoning Dataset (1,449 variants, 37 unique diseases), Variant Effect Prediction of Coding Sequences (50,083 core variant entries), and Variant Effect Prediction of Coding Non-SNVs (36,088 core non-SNV entries). On the KEGG-derived reasoning benchmark, the Evo2+Qwen3-4B model achieves 97.24% accuracy and an 86.30% F1-score. For variant effect prediction (VEP) tasks, the Evo2+Qwen3-4B model attains 80.21% accuracy for coding variants and 88.20% accuracy for non-SNV classification, significantly outperforming DNA-only and LLM-only baselines across all tasks.",
      },
      case_study: {
        image: "./static/images/case_study.png",
        text: "To illustrate BioReason's interpretable reasoning capabilities, consider a case where it was queried about the biological effect of a PFN1 allele on chromosome 17, given the pathway context 'Actin(monomeric) // PFN1* // Actin(filamentous)'. BioReason correctly predicted Amyotrophic Lateral Sclerosis (ALS) as the resultant disease. Significantly, the model generated a plausible 10-step mechanistic rationale, initiating by identifying a specific C>G substitution in the PFN1 gene. Its reasoning then connected this variant to profilin-1 dysfunction, impaired actin dynamics critical for cytoskeletal integrity, subsequent disruption of axonal transport in motor neurons, and finally, the motor neuron degeneration characteristic of ALS.",
      },
      dataset: {
        image: "./static/images/dataset.png",
        text: "We curated three comprehensive datasets for training and evaluation: a novel KEGG-derived biological reasoning dataset (1,449 entries) that elucidates mechanistic connections between genetic variants and disease phenotypes, a Variant Effect Prediction dataset for coding sequences (50,083 entries), and a dataset for coding non-SNVs (36,088 entries). The KEGG dataset uses standardized symbolic notation to represent molecular networks including activation, inhibition, and regulatory interactions, while the VEP datasets focus on pathogenic/benign classification and disease phenotype prediction across diverse genomic variants.",
      },
    },
    conclusion: "BioReason advances computational biology by seamlessly integrating high-capacity DNA sequence encoders with the flexible reasoning of large language models, yielding a unified framework that excels at both mechanistic pathway inference and variant pathogenicity prediction. Across KEGG-derived reasoning tasks and VEP benchmarks, our DNA–LLM hybrids consistently outperform models restricted to a single modality while generating transparent, stepwise explanations that facilitate expert validation. This tight multimodal fusion, further refined through reinforcement learning, not only boosts accuracy but also opens new avenues for interpretable genomic analysis.",
  },
  bibtex: `@misc{fallahpour2025bioreasonincentivizingmultimodalbiological,
      title={BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model}, 
      author={Adibvafa Fallahpour and Andrew Magnuson and Purav Gupta and Shihao Ma and Jack Naimer and Arnav Shah and Haonan Duan and Omar Ibrahim and Hani Goodarzi and Chris J. Maddison and Bo Wang},
      year={2025},
      eprint={2505.23579},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.23579}, 
}`,
};
