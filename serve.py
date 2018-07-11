import tensorflow as tf
import run

def get_filename(data_dir, sensor_type):
    assert os.path.exists(data_dir), ('data file does not exist')
    return os.path.join(data_dir, 'predict_r{}'.format(sensor_type)) 

def decode(sequence):   
    return tf.decode_raw(sequence, run._DEFAULT_DECODE_DTYPE)

def transform(dataset, batch_size, decode_fn):
    # make the dataset prefetchable for parallellism
    dataset = dataset.prefetch(buffer_size=batch_size)
    
    # Parse the raw records into images and labels and batch them
    dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda v: decode(v),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))
    
    # prefetch one batch at a time
    dataset.prefetch(1)
    return dataset

def input_fn(data_dir, batch_size, sensor_type):
    filename = get_filename(data_dir, sensor_type)
    dataset = tf.data.FixedLengthRecordDataset(filename, run._RECORD_BYTES-1)
    
    return transform(
      dataset=dataset,
      batch_size=batch_size,
      parse_record_fn=decode,
    )   

def main(args):
    inputs = input_fn(args.data_dir, args.batch_size, args.sensor_type)
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.export_dir)
        predictor = tf.contrib.predictor.from_saved_model(args.export_dir)
        outputs = predictor(inputs)
        predictions = outputs['classes'][0]  

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument('--sensor_type', type=int, default=None, help='sensor type')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')  
    parser.add_argument('--data_dir', type=str, default=os.getcwd(), help='directory to read data from')
    parser.add_argument('--export_dir', type=str, default=os.getcwd(), help='directory to load trained model from')

    args = parser.parse_args()  
  
    main(args)