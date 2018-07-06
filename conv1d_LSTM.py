import tensorflow as tf

DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.int8, tf.int16)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

class Conv1d_LSTM_Model():
    def __init__(self, layers, n_units, n_classes, dtype=DEFAULT_DTYPE):
        # layers: a list that specifies convolution-pooling architecture
        # list index indicate layer position in stack; 
        # a pooling layer is represented by a tuple: (pooling_type, kernel_size, strides)
        # a convolution layer is represented by a tuple: (filter_width, depth)
        self.layers = layers
        self.n_units = n_units
        self.n_classes = n_classes
        self.dtype = dtype
        
    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE, *args, **kwargs):
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        return getter(name, shape, dtype, *args, **kwargs)
    
    def _model_variable_scope(self):
        return tf.variable_scope('Conv1d_LSTM_model', custom_getter=self._custom_dtype_getter)
    
    def __call__(self, inputs, is_training):
        with self._model_variable_scope():
            # if layers arg is provided, transform the inputs according to 
            # the convolution or pooling specs in layers
            if self.layers is not None:              
                for l in self.layers:
                    if type(l[0]) is int:
                        inputs = tf.layers.conv1d(
                            inputs=inputs, kernel_size=l[0], filters=l[1], padding='same')
                    elif l[0] == 'MAX':
                        inputs = tf.layers.max_pooling1d(
                            inputs=inputs, pool_size=l[1], strides=l[2], padding='same')
                    elif l[0] == 'AVG':
                        inputs = tf.layers.average_pooling1d(
                            inputs=inputs, pool_size=l[1], strides=l[2], padding='same')
            if type(self.n_units) is list:
                multi_cells = [tf.contrib.rnn.LSTMCell(size) for size in self.n_units]
                cell = tf.contrib.rnn.MultiRNNCell(multi_cells)
            elif type(self.n_units) is int:
                cell = tf.contrib.rnn.LSTMCell(self.n_units) 
            outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, dtype=DEFAULT_DTYPE)
            outputs = tf.layers.dense(inputs=outputs[:,-1, :], units=self.n_classes)
            return outputs