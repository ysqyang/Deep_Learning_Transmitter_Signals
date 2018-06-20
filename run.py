import tensorflow as tf
import argparse
import os
import conv1d_LSTM

_LAYERS = None
_NUM_CHANNELS = 2
_NUM_CLASSES = 3
_NUM_UNITS = 100
_DEFAULT_SEQUENCE_BYTES = 
_RECORD_BYTES = _DEFAULT_SEQUENCE_BYTES + 1     # The record is the signal sequence plus a one-byte label

_NUM_SEQUENCES = {
    'train': ,
    'validation': ,
}

_DATASET_NAME = None

def get_filenames(is_training, data_dir):
    """Returns a list of filenames."""
    data_dir = os.path.join(data_dir, '')

    assert os.path.exists(data_dir), ('data file does not exist')

    if is_training:
        return [
            os.path.join(data_dir, 'train_batch_{}.bin'.format(i))
            for i in range(1, _NUM_DATA_FILES + 1)
        ]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, is_training):
    
    return signals, label

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
          lambda value: parse_record_fn(value, is_training),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))
    
    # prefetch one batch at a time
    dataset.prefetch(1)

    return dataset

def learning_schedule(batch_size, batch_denom, n_sequences, boundary_epochs, decay_rates):
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = n_examples / batch_size

    # multiply learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    # a global step means running an optimization op on a batch
    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn

def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.
    Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    Returns:
    A dataset that can be used for iteration.
    """
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

    return process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_NUM_IMAGES['train'],
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None
    )

def conv1d_LSTM_model_fn(features, labels, mode, 
                         layers, n_units, n_classes, 
                         regularization_const, learning_rate_fn, momentum,
                         data_format, loss_scale, dtype=Conv1d_LSTM_model.DEFAULT_DTYPE):
    """
    Args:
    features: tensor representing input sequences
    labels: tensor representing class labels for all input sequences
    mode: current estimator mode; 
    model_class: TensorFlow model class. 
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

    tf.summary.image('signals', features, max_outputs=6)
    features = tf.cast(features, dtype)
    model = conv1d_LSTM.Conv1d_LSTM_Model(layers=layers, n_units=n_units, n_classes=_NUM_CLASSES, dtype=dtype)
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
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add regularization term to loss.
    l2_loss = regularization_const * tf.add_n(
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = learning_rate_fn(global_step)
        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)

        if loss_scale != 1:
            #multiply by loss_scale to avoid underflow
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            # scale the gradients back to original before passing them to optimizer.
            unscaled_grad_vars = [(grad / loss_scale, var) for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
        else:
            minimize_op = optimizer.minimize(loss, global_step)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, 
                                      train_op=train_op, eval_metric_ops=metrics)

def main(args, model_function, input_function, shape=None):
    """
    Args:
    args: parsed flags.
    model_function: the function that instantiates the Model and builds the
    ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
    dataset that the estimator can train on. This will be wrapped with
    all the relevant flags for running and passed to estimator.
    shape: list of ints representing the shape of the images used for training.
    This is only used if args.export_dir is passed.
    """
    classifier = tf.estimator.Estimator(model_fn=model_function, model_dir=args.model_dir, 
                                        params={
                                            'layers': _LAYERS,
                                            'n_units': _NUM_UNITS,
                                            'n_classes': _NUM_CLASSES, 
                                            'regularization_const': 1e-4,
                                            'momentum': 0.9, 
                                            'data_format': args.data_format,
                                            'loss_scale': args.loss_scale,
                                            'learning_rate_fn': learning_schedule(batch_size=args.batch_size, 
                                                                                  batch_denom=128,
                                                                                  n_sequences=_NUM_SEQUENCES['train'], 
                                                                                  boundary_epochs=[100, 150, 200],
                                                                                  decay_rates=[1, 0.1, 0.01, 0.001]), 
                                            })

    batch_size, batch_denom, n_examples, boundary_epochs, decay_rates

    def input_fn_train():
        return input_function(is_training=True, data_dir=args.data_dir,
                              batch_size=args.batch_size, num_epochs=args.epochs_between_evals)

    def input_fn_eval():
        return input_function(is_training=False, data_dir=args.data_dir,
                              batch_size=args.batch_size, num_epochs=1)

    total_training_cycle = args.train_epochs // args.epochs_between_evals
    
    # training and evaluating
    for cycle_index in range(total_training_cycle):
        tf.logging.info('Starting a training cycle: {}/{}'.format(cycle_index, total_training_cycle))

        classifier.train(input_fn=input_fn_train, max_steps=args.max_train_steps)

        tf.logging.info('Starting to evaluate.')

        # args.max_train_steps is generally associated with testing and
        # profiling. As a result it is frequently called with synthetic data, which
        # will iterate forever. Passing steps=args.max_train_steps allows the
        # eval (which is generally unimportant in those circumstances) to terminate.
        # Note that eval will run for max_train_steps each loop, regardless of the
        # global_step count.
        eval_results = classifier.evaluate(input_fn=input_fn_eval, steps=args.max_train_steps)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  

    #parser.add_argument('--reg', type=float, default=1e-4, help='regularization constant')
    #parser.add_argument('--momentum', type=float, default=0.9, help='momentum used in momentum optimizer')
    parser.add_argument('--data_format', type=str, default='channels_last', help='data format of input features')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--training_epochs', type=int, default=250, help='number of training epochs')
    parser.add_argument('--epochs_between_evals', type=int, default=10, help='number of epochs between successive evaluations')
    parser.add_argument('--max_train_steps', type=int, default=10000, help='maxumum number of training steps'
    parser.add_argument('--loss_scale', type=int, default=1, help='scaling factor for loss')   
    parser.add_argument('--data_dir', type=str, default=None, help='directory to read data from')
    parser.add_argument('--model_dir', type=int, default=None, help='directory to save model parameters to')
    
    args = parser.parse_args()  
  
    main(args, conv1d_LSTM_model_fn, input_fn)