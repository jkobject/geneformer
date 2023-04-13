---
datasets: ctheodoris/Genecorpus-30M
---
# Geneformer
Geneformer is a transformer model pretrained on a large-scale corpus of ~30 million single cell transcriptomes to enable context-aware predictions in settings with limited data in network biology. 

<!---
See [our manuscript](manuscript_link) for details.
--->

# Model Description
Geneformer is transformer model pretrained on a [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M), a pretraining corpus comprised of ~30 million single cell transcriptomes from a broad range of human tissues. Each single cell’s transcriptome is presented to the model as a rank value encoding where genes are ranked by their expression in that cell normalized by their expression across the entire Genecorpus-30M. The rank value encoding provides a nonparametric representation of that cell’s transcriptome and takes advantage of the many observations of each gene’s expression across Genecorpus-30M to prioritize genes that distinguish cell state. Specifically, this method will deprioritize ubiquitously highly-expressed housekeeping genes by normalizing them to a lower rank. Conversely, genes such as transcription factors that may be lowly expressed when they are expressed but highly distinguish cell state will move to a higher rank within the encoding. Furthermore, this rank-based approach may be more robust against technical artifacts that may systematically bias the absolute transcript counts value while the overall relative ranking of genes within each cell remains more stable. 

The rank value encoding of each single cell’s transcriptome then proceeds through six transformer encoder units. Pretraining was accomplished using a masked learning objective where 15% of the genes within each transcriptome were masked and the model was trained to predict which gene should be within each masked position in that specific cell state using the context of the remaining unmasked genes. A major strength of this approach is that it is entirely self-supervised and can be accomplished on completely unlabeled data, which allows the inclusion of large amounts of training data without being restricted to samples with accompanying labels.

<!---We detail applications and results in [our manuscript](manuscript_link). 
--->
During pretraining, Geneformer gained a fundamental understanding of network dynamics, encoding network hierarchy in the model’s attention weights in a completely self-supervised manner. Fine-tuning Geneformer towards a diverse panel of downstream tasks relevant to chromatin and network dynamics using limited task-specific data demonstrated that Geneformer consistently boosted predictive accuracy. Applied to disease modeling with limited patient data, Geneformer identified candidate therapeutic targets. Overall, Geneformer represents an invaluable pretrained model from which fine-tuning towards a broad range of downstream applications can be pursued to accelerate discovery of key network regulators and candidate therapeutic targets.

# Application
The pretrained Geneformer model can be used directly, for example for in silico deletion analysis, or by fine-tuning towards the relevant downstream task, such as gene or cell state classification.

# Installation
In addition to the pretrained model, contained herein are functions for tokenizing and collating data specific to single cell transcriptomics. To install:

```bash
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

For usage, see [examples](https://huggingface.co/ctheodoris/Geneformer/tree/main/examples) for pretraining and fine-tuning. Please note that GPU resources are required for efficient usage of Geneformer.