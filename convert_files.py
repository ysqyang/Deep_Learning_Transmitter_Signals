'''
script for converting original binary data files into one
binary file in which each record consists of one byte of label 
followed by a fixed number of bytes representing the sequence 
'''
import argparse
import os
import numpy as np
import run

_DISTANCES = [5, 10, 15, 20]

def convert(sensor_type, data_dir, out_file_name, num_sequences_per_file, mode):   
    with open(out_file_name, mode='wb') as out_file:
        for i in range(run._NUM_CLASSES):
            for d in _DISTANCES:
                file_path = os.path.join(data_dir, 'data', str(i), 't{}_r{}_{}m'.format(i+1, sensor_type, d))
                file_size = os.path.getsize(file_path)
                with open(file_path, mode='rb') as f:
                    for _ in range(num_sequences_per_file):
                        f.seek(2*np.random.randint(0, (file_size-run._RECORD_BYTES+1)//2))
                        sequence = f.read(run._RECORD_BYTES-1)
                        # add label to the front of each extracted sequence 
                        if mode in {'train', 'validate'}:
                            out_file.write(i.to_bytes(1, byteorder='little'))
                        out_file.write(sequence)  

    out_path = os.path.join(data_dir, out_file_name)
    print(os.path.getsize(out_path))

def main(args):
    if args.sensor_type is None:
        print('Please specify sensor type')
        return
    convert(args.sensor_type, args.data_dir, args.out_file_name, args.num_sequences_per_file, args.mode)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_type', type=int, default=None, help='sensor type')
    parser.add_argument('--mode', type=str, default='train', help='mode, must be train, validate or predict')
    parser.add_argument('--data_dir', type=str, default=os.getcwd(), help='data file directory')
    parser.add_argument('--out_file_name', type=str, default='extracted_data', help='output file name')
    parser.add_argument('--num_sequences_per_file', type=int, default=5000, help='number of sequences to extract from each data file')

    args = parser.parse_args()
    main(args)