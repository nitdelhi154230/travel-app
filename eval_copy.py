#! /usr/bin/env python
import pandas as pd
from pathlib import Path
import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import re
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", '', "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
#MoU is signed
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
df = pd.read_csv(r'all_data_1.csv')
df = df[pd.notnull(df['text'])]
col = ['text', 'type']
Df = df[col]
df.columns = ['text', 'label']
df = df.dropna(how='any',axis=0)
# Generate labels
positive_examples = [s.strip() for s in (df.loc[df['label'] == 0,'text'])]
negative_examples = [s.strip() for s in (df.loc[df['label'] == 1,'text'])]
#del df
# Split by words
x_raw = positive_examples + negative_examples
x_raw = [clean_str(sent) for sent in x_raw]
# Generate labels
positive_labels = [[0, 1] for _ in positive_examples]
negative_labels = [[1, 0] for _ in negative_examples]
y_test = np.concatenate([positive_labels, negative_labels], 0)
    #x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
y_test = np.argmax(y_test, axis=1)

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)
f2=(os.path.abspath(os.path.join((os.path.abspath(os.getcwd())+FLAGS.checkpoint_dir), "..","vocab")))
print(f2)
#f1=Path(os.path.abspath(os.getcwd())+FLAGS.checkpoint_dir)
#f2=Path(f1.parent)
#f1=f1.parent()
#print(f1)
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
print("Vocab "+vocab_path)
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

TP=0
TN=0
FP=0
FN=0
print(all_predictions)
print(y_test)
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
	csv.writer(f).writerows(predictions_human_readable)# Print accuracy if y_test is defined
if y_test is not None:
    for i in range(0,y_test.size):
        if((all_predictions[i]==y_test[i] and y_test[i]==0)):
            TP=TP+1
        if((all_predictions[i]==y_test[i] and y_test[i]==1)):
            TN=TN+1
        if((all_predictions[i]!=y_test[i] and y_test[i]==0)):
            FP=FP+1
        if((all_predictions[i]!=y_test[i] and y_test[i]==1)):
            FN=FN+1
            print("FN")
            print(FN)
            print("i ")
            print(i)
#            print(df[i][text])

   # TN=(sum(all_predictonss==y_test and y_test==1))    
   # FP=sum(all_predictions!=y_test and y_test==0)
    #      # FP=FP+1
   # FN=sum(all_predictions!=y_test and y_test==1)
   #        FN=FN+1
    print("TP TN FP FN")
    print(TP,TN,FP,FN)
#TRUE
        
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
print("Precision")
precision=(TP)/((TP+FP))
print(precision)
print("Recall")
recall=(TP)/((TP+FN))
print(recall)
print("F1 score")
f1=2*precision*recall
f1=f1/(precision+recall)
print(f1)
print("accuracy")
accuracy=TP+TN
accuracy=accuracy/(TP+TN+FP+FN)
print(accuracy)
print("Specificity")
specificity=(TN)/(TN+FP)
print(specificity)

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
