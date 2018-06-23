'''
script for converting original binary data files into one
binary file in which each record consists of one byte of label 
followed by a fixed number of bytes representing the sequence 
'''
import argparse
import os
import run

def convert(data_dir, out_file_name, num_sequences_per_file):   
    with open(out_file_name, mode='wb') as out_file:
        for i in range(run._NUM_CLASSES):
            for j in range(1, run._NUM_DATA_FILES[i]+1):
                file_path = os.path.join(data_dir, str(i), '{}.ip'.format(j))
                '''
                if data file does not contain enough bytes to extract
                num_sequences_per_file sequences of _DEFAULT_SEQUENCE_BYTES
                each, change num_sequences_per_file to fit the data file size 
                '''
                file_size = os.path.get_size(file_path)
                if file_size < num_sequences_per_file * run._DEFAULT_SEQUENCE_BYTES:
                    num_sequences_per_file = file_size // run._DEFAULT_SEQUENCE_BYTES 
                with open(file_path, mode='rb') as f:
                    for _ in num_sequences_per_file:
                        sequence = f.read(run._DEFAULT_SEQUENCE_BYTES)
                        # add label to the front of each extracted sequence 
                        out_file.write(i.to_bytes(1, byteorder='little'))
                        out_file.write(sequence)

def main(args):
    convert(args.data_dir, args.out_file_name, args.num_sequences_per_file)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='~', help='data file directory')
    parser.add_argument('--out_file_name', type=str, default='all_data', help='output file name')
    parser.add_argument('--num_sequences_per_file', type=int, default='', help='number of sequences to extract from each data file')
    args = parser.parse_args()
    main(args)