import sys, os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

def to_tensorboard(model, output_path, name, labels=None):
    meta_file = name + ".tsv"
    placeholder = np.zeros((len(model.wv.index2word), model.wv.vector_size))

    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        file_metadata.write("Text\tFrequency\n".encode('utf-8'))
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            # if word == '':
            #     print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
            #     file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            # else:
            file_metadata.write("{0}".format(word).encode('utf-8'))
            if labels is not None:
                file_metadata.write("\t{}".format(labels[i]).encode('utf-8'))
            file_metadata.write(b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = name)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = name
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, name + '.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))
