# Ancestry Inference in Low Dimensional Space Using Autoencoders

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

If you would like to generate new data, or use data from other chromosomes:
1. Download the data from [The 1000 Genomes Project](https://www.internationalgenome.org/).
2. More specifically, SNP data in the .vcf format is supported, which can be downloaded [here](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/).

(Note that the chromosome file are heavily compressed. Each \.vcf file may take up >50GB of disk space!)
3. Use the various functions in `parse_vcf.py` to parse and filter the \.vcf file and randomly sample SNPs.
4. After generating your `train.csv` and `test.csv`, the model can be run on the new data.

If you would like to modify or add new autoencoder models, you can do so in the `auto_encoder.py` file.