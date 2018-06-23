'''
script for converting original binary data files into one
binary file in which each record consists of one byte of label 
followed by a fixed number of bytes representing the sequence 
'''
import argparse
import os
import run

def convert(data_dir, out_file_name):   
    with open(out_file_name, mode='wb') as out_file:
    	for i in range(run._NUM_CLASSES):
    		for j in range(1, run._NUM_DATA_FILES[i]+1):
    			file_path = os.path.join(data_dir, str(i), '{}.ip'.format(j))
    			with open(file_path, mode='rb') as f:
    				sequence = f.read(run._DEFAULT_SEQUENCE_BYTES)
    				out_file.write(i.to_bytes(1, byteorder='little'))
    				out_file.write(sequence)

def main(args):
	convert(args.data_dir, args.out_file_name)

if __name__ == '__main__':  
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='~', help='data file directory')
	parser.add_argument('--out_file_name', type=str, default=, help='output file name')
	args = parser.parse_args()
	main(args)