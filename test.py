# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
plt.style.use('ggplot')
slim = tf.contrib.slim
import data_utils
import tfmpl
import io
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import PIL.Image

log_eval = './logs'

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Create log_dir for evaluation information
if not os.path.exists(log_eval):
    os.mkdir(log_eval)

labels_to_name = {0:"",1:"INFO",2:"DEBUG",3:"ERROR"}

# File paths
tf.flags.DEFINE_string('test_data_file', None, 'Test data file path')
tf.flags.DEFINE_string('run_dir', None, 'Restore the model from this run')
tf.flags.DEFINE_string('checkpoint', None, 'Restore the graph from this checkpoint')

# Test batch size
tf.flags.DEFINE_integer('batch_size', 64, 'Test batch size')
FLAGS = tf.app.flags.FLAGS

# Restore parameters
with open(os.path.join(FLAGS.run_dir, 'params.pkl'), 'rb') as f:
    params = pkl.load(f, encoding='bytes')

# Restore vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(FLAGS.run_dir, 'vocab'))

# Load test data
data, labels, lengths, _ = data_utils.load_data(file_path=FLAGS.test_data_file,
                                                 sw_path=params['stop_word_file'],
                                                 min_frequency=params['min_frequency'],
                                                 max_length=params['max_length'],
                                                 language=params['language'],
                                                 vocab_processor=vocab_processor,
                                                 shuffle=False)

def gen_plot():
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot([1,2])
    plt.title("test")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


# Restore graph
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    # Restore metagraph
    saver = tf.train.import_meta_graph('{}.meta'.format(os.path.join(FLAGS.run_dir, 'model', FLAGS.checkpoint)))
    # Restore weights
    saver.restore(sess, os.path.join(FLAGS.run_dir, 'model', FLAGS.checkpoint))

    # Get tensors
    input_x = graph.get_tensor_by_name('input_x:0')
    input_y = graph.get_tensor_by_name('input_y:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    predictions = graph.get_tensor_by_name('softmax/predictions:0')
    accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
    print("^^^^^^^^^^^^^^^^",type(input_x))
    # Generate batches
    batches = data_utils.batch_iter(data, labels, lengths, FLAGS.batch_size, 1)

    num_batches = int(len(data)/FLAGS.batch_size)
    all_predictions = []
    sum_accuracy = 0

    # Test
    for batch in batches:
        x_test, y_test, x_lengths = batch
        if params['clf'] == 'cnn':
            feed_dict = {input_x: x_test, input_y: y_test, keep_prob: 1.0}
            batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)
        else:
            batch_size = graph.get_tensor_by_name('batch_size:0')
            sequence_length = graph.get_tensor_by_name('sequence_length:0')
            feed_dict = {input_x: x_test, input_y: y_test, batch_size: FLAGS.batch_size, sequence_length: x_lengths, keep_prob: 1.0}

            batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)

        sum_accuracy += batch_accuracy
        all_predictions = np.concatenate([all_predictions, batch_predictions])

    final_accuracy = sum_accuracy / num_batches

# Print test accuracy
print('Test accuracy: {}'.format(final_accuracy))

# Save all predictions
with open(os.path.join(FLAGS.run_dir, 'predictions.csv'), 'w', encoding='utf-8', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['True class', 'Prediction'])
    count = 0
    range_num = 20
    for i in range(len(all_predictions)):
        csvwriter.writerow([labels[i], all_predictions[i]])
    for ii in range(range_num):
        label, prediction = labels[ii], all_predictions[ii]
        # print("^^^^^^^^^^",label, prediction, "^^^^^^^^^^")
        prediction_name, label_name = labels_to_name[prediction], labels_to_name[label]
        # print("^^^^^^^^^^", prediction_name, label_name, "^^^^^^^^^^")
        text = 'Predicted: %s \t Actual: %s' %(prediction_name, label_name)
        print (text,"\n")
        if(prediction_name==label_name):
            count = count + 1
    differ = range_num - count
    diff_text = '\n\nTrue Positives: %s \t False Positives: %s' %(differ, count)
    print (diff_text)
    #Set up the plot and hide axes
    plt.plot(differ,count,'r*')
    plt.title(diff_text)
    # plt.show()

    print('Predictions saved to {}'.format(os.path.join(FLAGS.run_dir, 'predictions.csv')))

