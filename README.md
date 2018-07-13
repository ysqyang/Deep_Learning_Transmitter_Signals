# Classifying-transmitter-signal-sequences-with-deep-learning

Model

The model is defined in conv_1d_LSTM.py. This model features a recurrent neural network (RNN) that consists of long short-term memory (LSTM) units, with a single fully connected layer at the top and optional 1D convolution and pooling layers at the bottom. The convolution and pooling layers can be added through the _LAYERS argument using the format described in the comment. 

Converting data files for training purposes

The script convert_files.py converts raw binary data files into a format that can be readily used for training and evaluation purposes. The program has the following command line arguments:

--sensor_type: sensor type, must be 0 or 1
--out_file_name: name of converted file
--data_dir: directory in which the original data files are located
--num_sequences_per_file: the number of sequences to extract from each file

Once the arguments are specified, the program extracts args.num_sequences_per_file sequences from each file at random offsets and writes them to the output file. Eqch extracted sequence has a fixed length specified by a constant from run.py and represents a training instance. A label (i.e., transmitter type 0 or 1) is added at the beginning of each sequence written to facilitate subsequent training and evaluation.

Training and evaluation

The script run.py trains and evaluates the model defined in conv_1d_LSTM.py using the files generated by convert_files.py. The program uses the tf.data.Dataset for creating input pipelines and the tf.estimator.Estimator classes for training and evaluating. The script requires the following command line arguments:

--sensor_type: sensor type, must be 0 or 1
--data_format: "channels_first" or "channels_last", defaults to "channels_last"
--batch_size: 
