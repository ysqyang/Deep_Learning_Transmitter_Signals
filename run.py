import tensorflow as tf
import argparse
import os
import conv1d_LSTM
import numpy as np

_LAYERS = None
_NUM_CHANNELS = 2
_NUM_CLASSES = 2
_NUM_UNITS = 200
_DEFAULT_DECODE_DTYPE = tf.uint8
_DEFAULT_SEQUENCE_LENGTH = 1000
'''
The record is the signal sequence plus a one-byte label.
Each sequence has _DEFAULT_SEQUENCE_LENGTH time points, 
each of which consists of 2 points  
'''
_RECORD_BYTES = _DEFAULT_SEQUENCE_LENGTH*2 + 1     
_NUM_SEQUENCES = {
    'train': 40000,
    'validate': 10000
}

def get_filename(is_training, data_dir, sensor_type):
    assert os.path.exists(data_dir), ('data file does not exist')

    if is_training:
        return os.path.join(data_dir, 'train_r{}'.format(sensor_type)) 
    else:
        return os.path.join(data_dir, 'validate_r{}'.format(sensor_type))

def parse_record(raw_record):   
    record = tf.decode_raw(raw_record, _DEFAULT_DECODE_DTYPE)
    label = tf.cast(record[0], tf.int32)
    sequence = tf.reshape(record[1:], (_DEFAULT_SEQUENCE_LENGTH, _NUM_CHANNELS))
    return sequence, label

def preprocess(sequence, is_training):
    """Preprocess a signal sequence"""
    pass

def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer, parse_record_fn, num_epochs):
    # make the dataset prefetchable for parallellism
    dataset = dataset.prefetch(buffer_size=batch_size)
    
    # shuffle dataset
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    
    # repeat shuffled dataset for multi-epoch training
    dataset = dataset.repeat(num_epochs)

    # Parse the raw records into images and labels and batch them
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda v: parse_record_fn(v),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))
    
    # prefetch one batch at a time
    dataset.prefetch(1)
    return dataset

def input_fn(is_training, data_dir, batch_size, sensor_type, num_epochs=1):
    filename = get_filename(is_training, data_dir, sensor_type)
    dataset = tf.data.FixedLengthRecordDataset(filename, _RECORD_BYTES)
    
    return process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_NUM_SEQUENCES['train'],
      parse_record_fn=parse_record,
      num_epochs=num_epochs
    )   

def build_tensor_serving_input_receiver_fn(shape, dtype=tf.float32, batch_size=None):
    """Returns a input_receiver_fn that can be used during serving.
    Args:
    shape: list representing target size of a single example.
    dtype: the expected datatype for the input example
    batch_size: number of input tensors that will be passed for prediction
    Returns:
    A function that itself returns a TensorServingInputReceiver.
    """  
    def serving_input_receiver_fn():
        # Prep a placeholder where the input example will be fed in
        features = tf.placeholder(
            dtype=dtype, shape=[batch_size] + shape, name='input')

        return tf.estimator.export.TensorServingInputReceiver(
            features=features, receiver_tensors=features)

    return serving_input_receiver_fn

def learning_schedule(batch_size, batch_denom, n_sequences, boundary_epochs, decay_rates):
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = n_sequences / batch_size

    # multiply learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    # a global step means running an optimization op on a batch
    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn

def conv1d_LSTM_model_fn(features, labels, mode, params):
    """
    Args:
    features: tensor representing input sequences
    labels: tensor representing class labels for all input sequences
    mode: current estimator mode;  
    regularization_const: regularization constant used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    loss_scale: The factor to scale the loss for numerical stability. 
    dtype: the TensorFlow dtype to use for calculations.
    Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
    """
    features = tf.cast(features, params['dtype'])
    #.summary.audio(name='signals', tensor=features, sample_rate=100)
    model = conv1d_LSTM.Conv1d_LSTM_Model(layers=params['layers'], n_units=params['n_units'],
                                          n_classes=params['n_classes'], dtype=params['dtype'])
    
    logits = model(features, mode==tf.estimator.ModeKeys.TRAIN)
    logits = tf.cast(logits, tf.float32)

    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    # cross entropy part
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add regularization term to loss.
    l2_loss = params['regularization_const'] * tf.add_n(
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        #learning_rate = params['learning_rate_fn'](global_step)
        # Create a tensor named learning_rate for logging purposes
        #tf.identity(learning_rate, name='learning_rate')
        #tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.AdamOptimizer()

        if params['loss_scale'] != 1:
            #multiply by loss_scale to avoid underflow
            scaled_grad_vars = optimizer.compute_gradients(loss * params['loss_scale'])

            # scale the gradients back to original before passing them to optimizer.
            unscaled_grad_vars = [(grad / params['loss_scale'], var) for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            minimize_op = optimizer.minimize(loss, global_step)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, 
                                      train_op=train_op, eval_metric_ops=metrics)

def main(args, model_function, input_function, shape):
    """
    Args:
    args: parsed flags.
    model_function: the function that instantiates the Model and builds the
    ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
    dataset that the estimator can train on. This will be wrapped with
    all the relevant flags for running and passed to estimator.
    shape: list of ints representing the shape of the inputs used for training.
    """
    if args.sensor_type not in range(1, _NUM_CLASSES+1):
        print('Sensor type must be between 1 and {}'.format(_NUM_CLASSES))
        return

    model_dir = 'model_r{}'.format(args.sensor_type)
    export_dir = 'export_r{}'.format(args.sensor_type)
    try:
        os.makedirs(model_dir)
    except:
        print('model directory already exists. Please specify a different directory')
        return

    try:
        os.makedirs(export_dir)
    except:
        print('export directory already exists. Please specify a different directory')
        return
    
    classifier = tf.estimator.Estimator(model_fn=model_function, 
                                        model_dir=model_dir, 
                                        params={
                                            'layers': _LAYERS,
                                            'n_units': _NUM_UNITS,
                                            'n_classes': _NUM_CLASSES, 
                                            'regularization_const': 1e-4,
                                            #'momentum': 0.9,
                                            'data_format': args.data_format,
                                            'loss_scale': args.loss_scale,
                                            'dtype': conv1d_LSTM.DEFAULT_DTYPE
                                            #'learning_rate_fn': learning_schedule(batch_size=args.batch_size, 
                                            #                                      batch_denom=128,
                                            #                                      n_sequences=_NUM_SEQUENCES['train'], 
                                            #                                      boundary_epochs=[5, 15, 20],
                                            #                                      decay_rates=[1, 0.1, 0.01, 0.001]),            
                                            })

    def input_fn_train():
        return input_function(is_training=True, data_dir=args.data_dir,
                              batch_size=args.batch_size, sensor_type=args.sensor_type,
                              num_epochs=args.epochs_between_evals)

    def input_fn_eval():
        return input_function(is_training=False, data_dir=args.data_dir,
                              batch_size=args.batch_size, sensor_type=args.sensor_type,
                              num_epochs=1)

    total_training_cycle = args.train_epochs // args.epochs_between_evals
    
    # training and evaluating
    for cycle_index in range(total_training_cycle):
        tf.logging.info('Starting a training cycle: {}/{}'.format(cycle_index, total_training_cycle))
        #print('starting to train')
        classifier.train(input_fn=input_fn_train, max_steps=args.max_train_steps)
        tf.logging.info('Starting to evaluate.')
        eval_results = classifier.evaluate(input_fn=input_fn_eval, steps=args.max_train_steps)
   
    # Exports a saved model for the given classifier.
    input_receiver_fn = build_tensor_serving_input_receiver_fn(shape)
    classifier.export_savedmodel(export_dir, input_receiver_fn)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument('sensor_type', type=int, default=None, help='sensor type')
    parser.add_argument('--data_format', type=str, default='channels_last', help='data format of input features')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--train_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--epochs_between_evals', type=int, default=10, help='number of epochs between evaluations')
    parser.add_argument('--max_train_steps', type=int, default=20000, help='maxumum number of training steps')
    parser.add_argument('--loss_scale', type=int, default=1, help='scaling factor for loss')   
    parser.add_argument('--data_dir', type=str, default=os.getcwd(), help='directory to read data from')

    args = parser.parse_args()  
  
    main(args, conv1d_LSTM_model_fn, input_fn, shape=[_DEFAULT_SEQUENCE_LENGTH, _NUM_CHANNELS])
