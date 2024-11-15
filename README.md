---
datasets: ctheodoris/Genecorpus-30M
license: apache-2.0
tags:
- single-cell
- genomics
---
# Geneformer
Geneformer is a foundational transformer model pretrained on a large-scale corpus of single cell transcriptomes to enable context-aware predictions in settings with limited data in network biology.

- See [our manuscript](https://rdcu.be/ddrx0) for details of the original model trained on ~30 million transcriptomes in June 2021 and the initial report of our in silico perturbation and cell and gene classification strategies.
- See [our manuscript](https://www.biorxiv.org/content/10.1101/2024.08.16.608180v1.full.pdf) for details of the expanded model trained on ~95 million transcriptomes in April 2024 and our continual learning, multitask learning, and quantization strategies.
- See [geneformer.readthedocs.io](https://geneformer.readthedocs.io) for documentation.

# Model Description
Geneformer is a foundational transformer model pretrained on a large-scale corpus of single cell transcriptomes representing a broad range of human tissues. Geneformer was originally pretrained in June 2021 on [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M), a corpus comprised of ~30 million single cell transcriptomes. We excluded cells with high mutational burdens (e.g. malignant cells and immortalized cell lines) that could lead to substantial network rewiring without companion genome sequencing to facilitate interpretation. Then, in April 2024, Geneformer was pretrained on ~95 million non-cancer transcriptomes, followed by continual learning on ~14 million cancer transcriptomes to yield a cancer domain-tuned model. 

Each single cell’s transcriptome is presented to the model as a rank value encoding where genes are ranked by their expression in that cell scaled by their expression across the entire Genecorpus-30M. The rank value encoding provides a nonparametric representation of that cell’s transcriptome and takes advantage of the many observations of each gene’s expression across the pretraining corpus to prioritize genes that distinguish cell state. Specifically, this method will deprioritize ubiquitously highly-expressed housekeeping genes by scaling them to a lower rank. Conversely, genes such as transcription factors that may be lowly expressed when they are expressed but highly distinguish cell state will move to a higher rank within the encoding. Furthermore, this rank-based approach may be more robust against technical artifacts that may systematically bias the absolute transcript counts value while the overall relative ranking of genes within each cell remains more stable.

The rank value encoding of each single cell’s transcriptome then proceeds through N layers of transformer encoder units, where N varies dependent on the model size. Pretraining was accomplished using a masked learning objective where 15% of the genes within each transcriptome were masked and the model was trained to predict which gene should be within each masked position in that specific cell state using the context of the remaining unmasked genes. A major strength of this approach is that it is entirely self-supervised and can be accomplished on completely unlabeled data, which allows the inclusion of large amounts of training data without being restricted to samples with accompanying labels.

We detail applications and results in [our manuscript](https://rdcu.be/ddrx0).

During pretraining, Geneformer gained a fundamental understanding of network dynamics, encoding network hierarchy in the model’s attention weights in a completely self-supervised manner. With both zero-shot learning and fine-tuning with limited task-specific data, Geneformer consistently boosted predictive accuracy in a diverse panel of downstream tasks relevant to chromatin and network dynamics. In silico perturbation with zero-shot learning identified a novel transcription factor in cardiomyocytes that we experimentally validated to be critical to their ability to generate contractile force. In silico treatment with limited patient data revealed candidate therapeutic targets for cardiomyopathy that we experimentally validated to significantly improve the ability of cardiomyocytes to generate contractile force in an induced pluripotent stem cell (iPSC) model of the disease. Overall, Geneformer represents a foundational deep learning model pretrained on a large-scale corpus human single cell transcriptomes to gain a fundamental understanding of gene network dynamics that can now be democratized to a vast array of downstream tasks to accelerate discovery of key network regulators and candidate therapeutic targets.

The repository includes the following pretrained models:

L=layers\
M=millions of cells used for pretraining\
i=input size\
(pretraining date)

- GF-6L-30M-i2048 (June 2021)
- GF-12L-30M-i2048 (June 2021)
- GF-12L-95M-i4096 (April 2024)
- GF-20L-95M-i4096 (April 2024)

The current default model in the main directory of the repository is GF-12L-95M-i4096.

The repository also contains fined tuned models in the fine_tuned_models directory and the cancer-tuned model following continual learning on ~14 million cancer cells, GF-12L-95M-i4096_CLcancer.

# Application
The pretrained Geneformer model can be used directly for zero-shot learning, for example for in silico perturbation analysis, or by fine-tuning towards the relevant downstream task, such as gene or cell state classification.

Example applications demonstrated in [our manuscript](https://rdcu.be/ddrx0) include:

*Fine-tuning*:
- transcription factor dosage sensitivity
- chromatin dynamics (bivalently marked promoters)
- transcription factor regulatory range
- gene network centrality
- transcription factor targets
- cell type annotation
- batch integration
- cell state classification across differentiation
- disease classification
- in silico perturbation to determine disease-driving genes
- in silico treatment to determine candidate therapeutic targets

*Zero-shot learning*:
- batch integration
- gene context specificity
- in silico reprogramming
- in silico differentiation
- in silico perturbation to determine impact on cell state
- in silico perturbation to determine transcription factor targets
- in silico perturbation to determine transcription factor cooperativity

# Installation
In addition to the pretrained model, contained herein are functions for tokenizing and collating data specific to single cell transcriptomics, pretraining the model, fine-tuning the model, extracting and plotting cell embeddings, and performing in silico pertrubation with either the pretrained or fine-tuned models. To install (~20s):

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

For usage, see [examples](https://huggingface.co/ctheodoris/Geneformer/tree/main/examples) for:
- tokenizing transcriptomes
- pretraining
- hyperparameter tuning
- fine-tuning
- extracting and plotting cell embeddings
- in silico perturbation

Please note that the fine-tuning examples are meant to be generally applicable and the input datasets and labels will vary dependent on the downstream task. Example input files for a few of the downstream tasks demonstrated in the manuscript are located within the [example_input_files directory](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files) in the dataset repository, but these only represent a few example fine-tuning applications.

Please note that GPU resources are required for efficient usage of Geneformer. Additionally, we strongly recommend tuning hyperparameters for each downstream fine-tuning application as this can significantly boost predictive potential in the downstream task (e.g. max learning rate, learning schedule, number of layers to freeze, etc.).

# Citations
- C V Theodoris#, L Xiao, A Chopra, M D Chaffin, Z R Al Sayed, M C Hill, H Mantineo, E Brydon, Z Zeng, X S Liu, P T Ellinor#. Transfer learning enables predictions in network biology. _**Nature**_, 31 May 2023. (#co-corresponding authors)
- H Chen*, M S Venkatesh*, J Gomez Ortega, S V Mahesh, T Nandi, R Madduri, K Pelka†, C V Theodoris†#. Quantized multi-task learning for context-specific representations of gene network dynamics. _**bioRxiv**_, 19 Aug 2024. (*co-first authors, †co-senior authors, #corresponding author)