"""
Geneformer embedding extractor.

Usage:
  from geneformer import EmbExtractor
  embex = EmbExtractor(model_type="CellClassifier",
                       num_classes=3,
                       emb_mode="cell",
                       cell_emb_style="mean_pool",
                       filter_data={"cell_type":["cardiomyocyte"]},
                       max_ncells=1000,
                       max_ncells_to_plot=1000,
                       emb_layer=-1,
                       emb_label=["disease","cell_type"],
                       labels_to_plot=["disease","cell_type"],
                       forward_batch_size=100,
                       nproc=16,
                       summary_stat=None)
  embs = embex.extract_embs("path/to/model",
                            "path/to/input_data",
                            "path/to/output_directory",
                            "output_prefix")
  embex.plot_embs(embs=embs, 
                  plot_style="heatmap",
                  output_directory="path/to/output_directory",
                  output_prefix="output_prefix")
  
"""

# imports
import logging
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tdigest import TDigest
import scanpy as sc
import seaborn as sns
import torch
from collections import Counter
from pathlib import Path
from tqdm.notebook import trange
from transformers import BertForMaskedLM, BertForTokenClassification, BertForSequenceClassification

from .tokenizer import TOKEN_DICTIONARY_FILE

from .in_silico_perturber import downsample_and_sort, \
                                 gen_attention_mask, \
                                 get_model_input_size, \
                                 load_and_filter, \
                                 load_model, \
                                 mean_nonpadding_embs, \
                                 pad_tensor_list, \
                                 quant_layers

logger = logging.getLogger(__name__)

# extract embeddings
def get_embs(model,
             filtered_input_data,
             emb_mode,
             layer_to_quant,
             pad_token_id,
             forward_batch_size,
             summary_stat):
    
    model_input_size = get_model_input_size(model)
    total_batch_length = len(filtered_input_data)
    
    if summary_stat is None:
        embs_list = []
    elif summary_stat is not None:
        # test embedding extraction for example cell and extract # emb dims
        example = filtered_input_data.select([i for i in range(1)])
        example.set_format(type="torch")
        emb_dims = test_emb(model, example["input_ids"], layer_to_quant)
        # initiate tdigests for # of emb dims
        embs_tdigests = [TDigest() for _ in range(emb_dims)]

    for i in trange(0, total_batch_length, forward_batch_size):
        max_range = min(i+forward_batch_size, total_batch_length)

        minibatch = filtered_input_data.select([i for i in range(i, max_range)])
        max_len = max(minibatch["length"])
        original_lens = torch.tensor(minibatch["length"]).to("cuda")
        minibatch.set_format(type="torch")

        input_data_minibatch = minibatch["input_ids"]
        input_data_minibatch = pad_tensor_list(input_data_minibatch, 
                                               max_len, 
                                               pad_token_id, 
                                               model_input_size)
        
        with torch.no_grad():
            outputs = model(
                input_ids = input_data_minibatch.to("cuda"),
                attention_mask = gen_attention_mask(minibatch)
            )

        embs_i = outputs.hidden_states[layer_to_quant]
        
        if emb_mode == "cell":
            mean_embs = mean_nonpadding_embs(embs_i, original_lens)
            if summary_stat is None:
                embs_list += [mean_embs]
            elif summary_stat is not None:
                # update tdigests with current batch for each emb dim
                # note: tdigest batch update known to be slow so updating serially
                [embs_tdigests[j].update(mean_embs[i,j].item()) for i in range(mean_embs.size(0)) for j in range(emb_dims)]
            
        del outputs
        del minibatch
        del input_data_minibatch
        del embs_i
        del mean_embs
        torch.cuda.empty_cache()            
    
    if summary_stat is None:
        embs_stack = torch.cat(embs_list)
    # calculate summary stat embs from approximated tdigests
    elif summary_stat is not None:
        if summary_stat == "mean":
            summary_emb_list = [embs_tdigests[i].trimmed_mean(0,100) for i in range(emb_dims)]
        elif summary_stat == "median":
            summary_emb_list = [embs_tdigests[i].percentile(50) for i in range(emb_dims)]
        embs_stack = torch.tensor(summary_emb_list)

    return embs_stack

def test_emb(model, example, layer_to_quant):
    with torch.no_grad():
        outputs = model(
            input_ids = example.to("cuda")
        )

    embs_test = outputs.hidden_states[layer_to_quant]
    return embs_test.size()[2]

def label_embs(embs, downsampled_data, emb_labels):
    embs_df = pd.DataFrame(embs.cpu())
    if emb_labels is not None:
        for label in emb_labels:
            emb_label = downsampled_data[label]
            embs_df[label] = emb_label
    return embs_df

def plot_umap(embs_df, emb_dims, label, output_file, kwargs_dict):
    only_embs_df = embs_df.iloc[:,:emb_dims]
    only_embs_df.index = pd.RangeIndex(0, only_embs_df.shape[0], name=None).astype(str)
    only_embs_df.columns = pd.RangeIndex(0, only_embs_df.shape[1], name=None).astype(str)
    vars_dict = {"embs": only_embs_df.columns}
    obs_dict = {"cell_id": list(only_embs_df.index),
                f"{label}": list(embs_df[label])}
    adata = anndata.AnnData(X=only_embs_df, obs=obs_dict, var=vars_dict)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sns.set(rc={'figure.figsize':(10,10)}, font_scale=2.3)
    sns.set_style("white")
    default_kwargs_dict = {"palette":"Set2", "size":200}
    if kwargs_dict is not None:
        default_kwargs_dict.update(kwargs_dict)
        
    sc.pl.umap(adata, color=label, save=output_file, **default_kwargs_dict)

def gen_heatmap_class_colors(labels, df):
    pal = sns.cubehelix_palette(len(Counter(labels).keys()), light=0.9, dark=0.1, hue=1, reverse=True, start=1, rot=-2)
    lut = dict(zip(map(str, Counter(labels).keys()), pal))
    colors = pd.Series(labels, index=df.index).map(lut)
    return colors
    
def gen_heatmap_class_dict(classes, label_colors_series):
    class_color_dict_df = pd.DataFrame({"classes": classes, "color": label_colors_series})
    class_color_dict_df = class_color_dict_df.drop_duplicates(subset=["classes"])
    return dict(zip(class_color_dict_df["classes"],class_color_dict_df["color"]))
    
def make_colorbar(embs_df, label):

    labels = list(embs_df[label])
                  
    cell_type_colors = gen_heatmap_class_colors(labels, embs_df)
    label_colors = pd.DataFrame(cell_type_colors, columns=[label])

    for i,row in label_colors.iterrows():
        colors=row[0]
        if len(colors)!=3 or any(np.isnan(colors)):
            print(i,colors)

    label_colors.isna().sum()
    
    # create dictionary for colors and classes
    label_color_dict = gen_heatmap_class_dict(labels, label_colors[label])
    return label_colors, label_color_dict
    
def plot_heatmap(embs_df, emb_dims, label, output_file, kwargs_dict):
    sns.set_style("white")
    sns.set(font_scale=2)
    plt.figure(figsize=(15, 15), dpi=150)
    label_colors, label_color_dict = make_colorbar(embs_df, label)
    
    default_kwargs_dict = {"row_cluster": True,
                           "col_cluster": True,
                           "row_colors": label_colors,
                           "standard_scale":  1,
                           "linewidths": 0,
                           "xticklabels": False,
                           "yticklabels": False,
                           "figsize": (15,15),
                           "center": 0,
                           "cmap": "magma"}
    
    if kwargs_dict is not None:
        default_kwargs_dict.update(kwargs_dict)
    g = sns.clustermap(embs_df.iloc[:,0:emb_dims].apply(pd.to_numeric), **default_kwargs_dict)

    plt.setp(g.ax_row_colors.get_xmajorticklabels(), rotation=45, ha="right")

    for label_color in list(label_color_dict.keys()):
        g.ax_col_dendrogram.bar(0, 0, color=label_color_dict[label_color], label=label_color, linewidth=0)

        l1 = g.ax_col_dendrogram.legend(title=f"{label}", 
                                        loc="lower center", 
                                        ncol=4, 
                                        bbox_to_anchor=(0.5, 1), 
                                        facecolor="white")

    plt.savefig(output_file, bbox_inches='tight')

class EmbExtractor:
    valid_option_dict = {
        "model_type": {"Pretrained","GeneClassifier","CellClassifier"},
        "num_classes": {int},
        "emb_mode": {"cell","gene"},
        "cell_emb_style": {"mean_pool"},
        "filter_data": {None, dict},
        "max_ncells": {None, int},
        "emb_layer": {-1, 0},
        "emb_label": {None, list},
        "labels_to_plot": {None, list},
        "forward_batch_size": {int},
        "nproc": {int},
        "summary_stat": {None, "mean", "median"},
    }
    def __init__(
        self,
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        cell_emb_style="mean_pool",
        filter_data=None,
        max_ncells=1000,
        emb_layer=-1,
        emb_label=None,
        labels_to_plot=None,
        forward_batch_size=100,
        nproc=4,
        summary_stat=None,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize embedding extractor.

        Parameters
        ----------
        model_type : {"Pretrained","GeneClassifier","CellClassifier"}
            Whether model is the pretrained Geneformer or a fine-tuned gene or cell classifier.
        num_classes : int
            If model is a gene or cell classifier, specify number of classes it was trained to classify.
            For the pretrained Geneformer model, number of classes is 0 as it is not a classifier.
        emb_mode : {"cell","gene"}
            Whether to output cell or gene embeddings.
        cell_emb_style : "mean_pool"
            Method for summarizing cell embeddings.
            Currently only option is mean pooling of gene embeddings for given cell.
        filter_data : None, dict
            Default is to extract embeddings from all input data.
            Otherwise, dictionary specifying .dataset column name and list of values to filter by.
        max_ncells : None, int
            Maximum number of cells to extract embeddings from.
            Default is 1000 cells randomly sampled from input data.
            If None, will extract embeddings from all cells.
        emb_layer : {-1, 0}
            Embedding layer to extract.
            The last layer is most specifically weighted to optimize the given learning objective.
            Generally, it is best to extract the 2nd to last layer to get a more general representation.
            -1: 2nd to last layer
            0: last layer
        emb_label : None, list
            List of column name(s) in .dataset to add as labels to embedding output.
        labels_to_plot : None, list
            Cell labels to plot.
            Shown as color bar in heatmap.
            Shown as cell color in umap.
            Plotting umap requires labels to plot.
        forward_batch_size : int
            Batch size for forward pass.
        nproc : int
            Number of CPU processes to use.
        summary_stat : {None, "mean", "median"}
            If not None, outputs only approximated mean or median embedding of input data.
            Recommended if encountering memory constraints while generating goal embedding positions.
            Slower but more memory-efficient.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl ID:token).
        """

        self.model_type = model_type
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.cell_emb_style = cell_emb_style
        self.filter_data = filter_data
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.emb_label = emb_label
        self.labels_to_plot = labels_to_plot
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.summary_stat = summary_stat

        self.validate_options()

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        self.pad_token_id = self.gene_token_dict.get("<pad>")
        
        
    def validate_options(self):
        # first disallow options under development
        if self.emb_mode == "gene":
            logger.error(
                "Extraction and plotting of gene-level embeddings currently under development. " \
                "Current valid option for 'emb_mode': 'cell'"
            )
            raise
            
        # confirm arguments are within valid options and compatible with each other
        for attr_name,valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if type(attr_value) not in {list, dict}:
                if attr_value in valid_options:
                    continue
            valid_type = False
            for option in valid_options:
                if (option in [int,list,dict]) and isinstance(attr_value, option):
                    valid_type = True
                    break
            if valid_type:
                continue
            logger.error(
                f"Invalid option for {attr_name}. " \
                f"Valid options for {attr_name}: {valid_options}"
            )
            raise
        
        if self.filter_data is not None:
            for key,value in self.filter_data.items():
                if type(value) != list:
                    self.filter_data[key] = [value]
                    logger.warning(
                        "Values in filter_data dict must be lists. " \
                        f"Changing {key} value to list ([{value}]).")  
        
    def extract_embs(self, 
                     model_directory,
                     input_data_file,
                     output_directory,
                     output_prefix):
        """
        Extract embeddings from input data and save as results in output_directory.

        Parameters
        ----------
        model_directory : Path
            Path to directory containing model
        input_data_file : Path
            Path to directory containing .dataset inputs
        output_directory : Path
            Path to directory where embedding data will be saved as csv
        output_prefix : str
            Prefix for output file
        """

        filtered_input_data = load_and_filter(self.filter_data, self.nproc, input_data_file)
        downsampled_data = downsample_and_sort(filtered_input_data, self.max_ncells)
        model = load_model(self.model_type, self.num_classes, model_directory)
        layer_to_quant = quant_layers(model)+self.emb_layer
        embs = get_embs(model,
                        downsampled_data,
                        self.emb_mode,
                        layer_to_quant,
                        self.pad_token_id,
                        self.forward_batch_size,
                        self.summary_stat)
        
        if self.summary_stat is None:
            embs_df = label_embs(embs, downsampled_data, self.emb_label)
        elif self.summary_stat is not None:
            embs_df = pd.DataFrame(embs.cpu()).T

        # save embeddings to output_path
        output_path = (Path(output_directory) / output_prefix).with_suffix(".csv")
        embs_df.to_csv(output_path)

        return embs_df        
    
    def plot_embs(self,
                  embs, 
                  plot_style,
                  output_directory,
                  output_prefix,
                  max_ncells_to_plot=1000,
                  kwargs_dict=None):
        
        """
        Plot embeddings, coloring by provided labels.

        Parameters
        ----------
        embs : pandas.core.frame.DataFrame
            Pandas dataframe containing embeddings output from extract_embs
        plot_style : str
            Style of plot: "heatmap" or "umap"
        output_directory : Path
            Path to directory where plots will be saved as pdf
        output_prefix : str
            Prefix for output file
        max_ncells_to_plot : None, int
            Maximum number of cells to plot.
            Default is 1000 cells randomly sampled from embeddings.
            If None, will plot embeddings from all cells.
        kwargs_dict : dict
            Dictionary of kwargs to pass to plotting function.
        """
        
        if plot_style not in ["heatmap","umap"]:
            logger.error(
                "Invalid option for 'plot_style'. " \
                "Valid options: {'heatmap','umap'}"
            )
            raise
        
        if (plot_style == "umap") and (self.labels_to_plot is None):
            logger.error(
                "Plotting UMAP requires 'labels_to_plot'. "
            )
            raise
        
        if max_ncells_to_plot > self.max_ncells:
            max_ncells_to_plot = self.max_ncells
            logger.warning(
                "max_ncells_to_plot must be <= max_ncells. " \
                f"Changing max_ncells_to_plot to {self.max_ncells}.") 
        
        if (max_ncells_to_plot is not None) \
            and (max_ncells_to_plot < self.max_ncells):
            embs = embs.sample(max_ncells_to_plot, axis=0)
        
        if self.emb_label is None:
            label_len = 0
        else:
            label_len = len(self.emb_label)
        
        emb_dims = embs.shape[1] - label_len
        
        if self.emb_label is None:
            emb_labels = None
        else:
            emb_labels = embs.columns[emb_dims:]
        
        if plot_style == "umap":
            for label in self.labels_to_plot:
                if label not in emb_labels:
                    logger.warning(
                        f"Label {label} from labels_to_plot " \
                        f"not present in provided embeddings dataframe.")
                    continue
                output_prefix_label = "_" + output_prefix + f"_umap_{label}"
                output_file = (Path(output_directory) / output_prefix_label).with_suffix(".pdf")
                plot_umap(embs, emb_dims, label, output_prefix_label, kwargs_dict)
                
        if plot_style == "heatmap":
            for label in self.labels_to_plot:
                if label not in emb_labels:
                    logger.warning(
                        f"Label {label} from labels_to_plot " \
                        f"not present in provided embeddings dataframe.")
                    continue
                output_prefix_label = output_prefix + f"_heatmap_{label}"
                output_file = (Path(output_directory) / output_prefix_label).with_suffix(".pdf")
                plot_heatmap(embs, emb_dims, label, output_file, kwargs_dict)