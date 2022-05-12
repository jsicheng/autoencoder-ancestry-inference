import vcf
import pandas as pd
from tqdm import tqdm
import random
import itertools
import bisect
from sklearn.model_selection import train_test_split


def get_num_lines_and_header(infile):
    num_lines = 0
    num_header = 0
    with open(infile, 'r') as ifile:
        for line in ifile:
            num_lines += 1
            if line.startswith('#'):
                num_header += 1
    return num_lines, num_header


def filter_and_sample_vcf(infile, outfile, n_samples, num_lines=None, num_header=None):
    print("Getting number of lines...")
    if num_lines is None or num_header is None:
        num_lines, num_header = get_num_lines_and_header(infile)
    sample_list = sorted(random.sample(range(0, num_lines - num_header), n_samples))

    print("Filtering and sampling the .vcf file...")
    with open(infile, 'r') as reader, open(outfile, 'w') as writer:
        # write the header
        header_lines = list(itertools.islice(reader, num_header))
        for line in header_lines:
            writer.write(line)

        # write the sampled records
        record_index = 0
        n_resampled = 0
        for i, record in enumerate(tqdm(reader)):
            if record_index >= n_samples:
                break
            elif i < sample_list[record_index]:
                pass
            elif i == sample_list[record_index]:
                record_index += 1
                info = record.split('\t')[7].split(';')
                variant = [x for x in info if x.startswith('VT')][-1].split('=')
                multi_allelic_flag = [x for x in info if x.startswith('MULTI_ALLELIC')]
                if variant == "SNP" or not multi_allelic_flag:
                    writer.write(record)
                else:
                    new_record = random.randint(i + 1, num_lines - num_header - 1)
                    while new_record in sample_list[record_index:]:
                        new_record = random.randint(i + 1, num_lines - num_header - 1)
                    # print("Non-SNP variant sampled: {}, resampled: {}".format(variant, new_record))
                    n_samples += 1
                    n_resampled += 1
                    bisect.insort(sample_list, new_record)
        print("Resampled:", n_resampled)


def vcf_to_csv(vcf_file, csv_file, panel_file=None):
    vcf_reader = vcf.Reader(open(vcf_file, 'r'))
    samples = []
    for record in tqdm(vcf_reader):  # per SNP
        # data = {'sample': record.ID}  # not used for this .vcf
        data = {'sample': 'pos{}'.format(record.POS)}  # use position of SNP as name
        # if record.INFO['VT'][0] == 'SNP':  # use only SNPs
        for sample in record.samples:  # per individual
            if sample['GT'] == '0|0':
                data[sample.sample] = 0
            elif sample['GT'] == '0|1' or sample['GT'] == '1|0':
                data[sample.sample] = 0.5
            elif sample['GT'] == '1|1':
                data[sample.sample] = 1
            else:
                print("Non-biallelic SNP: {} INFO:".format(sample['GT']), record.INFO)
                data[sample.sample] = random.sample([0, 0.5, 0.5, 1], 1)
            # data[sample.sample] = sample['GT']  # sets individual's genotype
        samples.append(data)
    print("Writing to csv...")
    snps_df = pd.DataFrame(samples).set_index(['sample']).transpose()
    if panel_file is not None:
        # extract population information
        info_df = pd.read_csv(panel_file, sep='\t').set_index(["sample"])
        info_df.drop(info_df.columns[[2, 3, 4]], axis=1, inplace=True)
        # filter unannotated individuals and join
        snps_df = snps_df.join(info_df, how='outer')
    # write to csv
    snps_df.to_csv(csv_file)


if __name__ == "__main__":
    # # sample SNPs from vcf file
    # raw_vcf_file = "./data/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf"  # 6468347 Lines - 253 Comments = 6468094 Variants
    # sampled_vcf_file = "./data/chr1_sampled.vcf"
    #
    # filter_and_sample_vcf(infile=raw_vcf_file, outfile=sampled_vcf_file, n_samples=10000, num_lines=6468347, num_header=253)
    # print(get_num_lines_and_header(sampled_vcf_file))
    #
    # # convert to csv for pandas
    # vcf_to_csv(vcf_file=sampled_vcf_file, csv_file='./data/chr1_sampled_snps.csv', panel_file='./data/integrated_call_samples_v3.20130502.ALL.panel')
    #
    # # split into train/test
    train_df, test_df = train_test_split(pd.read_csv('./data/chr1_sampled_snps.csv', index_col=0), test_size=0.20)
    print("training_df shape", train_df.shape)
    print("testing_df shape", test_df.shape)
    train_df.to_csv('./data/chr1_train.csv')
    test_df.to_csv('./data/chr1_test.csv')
