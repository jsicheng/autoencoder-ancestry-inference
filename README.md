# Ancestry Inference in Low Dimensional Space Using Autoencoders

Read the paper [here](https://github.com/jsicheng/autoencoder-ancestry-inference/blob/main/Ancestry%20Inference%20in%20Low%20Dimensional%20Space%20Using%20Autoencoders.pdf).

The focus of this project is to explore the viability of using autoencoders for dimensionality reduction and ancestry inference. Convolutional, LSTM and GRU autoencoders were built and tested against PCA, UMAP, and vanilla autoencoders. The architecture of all autoencoders are listed in the paper.

Each dimensionality reduction method is tested on various classication models. The classification accuracies are presented in the table below:

<div align="center">

|                       | PCA    | UMAP     | AE    | CAE       | LSTMAE    | GRUAE     |
| --------------------- | ------ | ----     | ----- | -----     | --------- | --------- |
| Logistic Regression   | 0.888  | 0.936    | 0.934 | 0.858     | 0.946     | **0.966** |
| Random Forest         | 0.882  | 0.902    | 0.954 | **0.970** | **0.970** | 0.964     |
| SVM                   | 0.908  | 0.934    | 0.960 | **0.976** | 0.972     | 0.972     |
| MLP                   | 0.916  | 0.934    | 0.956 | **0.974** | 0.970     | **0.974** |

</div>

In two dimensions, the low dimensional mappings of the training data from each dimensionality reduction method produced interesting plots:

<table align="center">
<tr>
    <td> PCA </td>
    <td> UMAP </td>
    <td> AE </td>
</tr>
<tr>
    <td> <img src="/fig/pca_train.png" width="250" /> </td>
    <td> <img src="/fig/umap_train.png" width="250" /> </td>
    <td> <img src="/fig/ae_train.png" width="250" /> </td>
</tr>
<tr>
    <td> CAE </td>
    <td> LSTMAE </td>
    <td> GRUAE </td>
</tr>
<tr>
    <td> <img src="/fig/cae_train.png" width="250" /> </td>
    <td> <img src="/fig/lstmae_train.png" width="250" /> </td>
    <td> <img src="/fig/gruae_train.png" width="250" /> </td>
</tr>
</table>


<!-- | PCA | UMAP | AE |
| ![PCA](/fig/pca_train.png) | ![UMAP](/fig/umap_train.png) | ![AE](/fig/ae_train.png) |
| CAE | LSTMAE | GRUAE |
| ![CAE](/fig/cae_train.png) | ![LSTMAE](/fig/LSTMAE_train.png) | ![GRUAE](/fig/gruae_train.png) | -->

## Setup

To install and run this project, first clone the repository:

```sh
https://github.com/jsicheng/autoencoder-ancestry-inference.git
```

Then install the required python packages:

```sh
pip install -r requirements.txt
```

## Usage

To train and run all models:

```sh
python run_models.py
```

If you would like to sample new SNP data, or use SNP data from other chromosomes:
1. Download the data from [The 1000 Genomes Project](https://www.internationalgenome.org/).
2. SNP data in the .vcf format is supported, which can be downloaded [here](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/).\
(Note that the chromosome file are heavily compressed. Each \.vcf file may take up >50GB of disk space!)
3. Use the various functions in `parse_vcf.py` to parse and filter the \.vcf file and randomly sample SNPs.
4. After generating your `train.csv` and `test.csv`, the models can be run on the new data.

If you would like to modify or add new autoencoder models, you can do so in the `auto_encoder.py` file.