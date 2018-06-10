import numpy as np 
import gensim.models.word2vec as word2vec
import gensim.models.doc2vec as doc2vec
import gensim.test.utils
import pathlib
import sys, os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# %reload_ext autoreload
# %autoreload 2

# for in depth tracking of gensim models:
# import logging 
# logging.basicConfig(format='%(message)s', level=logging.INFO)

class Trainer:
    def __init__(self, save_dir, algorithm):
        self.save_dir = pathlib.Path(save_dir)
        self.algorithm = algorithm
    
    def __iter__(self):
        with self.save_dir.joinpath('line_documents.txt').open('r') as f:
            for i, line in enumerate(f):
                yield doc2vec.TaggedDocument(line.split(' '), [i])
        
    def train(self, **kwargs):
        arg_string = '_'.join( key + '=' + str(value) 
            for key, value in kwargs.items())
        print(self.algorithm + '_' + arg_string)
        
        if self.algorithm == 'word2vec':
            sentences = word2vec.PathLineSentences(self.save_dir.joinpath('line_sentences'))
            self.model = word2vec.Word2Vec(sentences, **kwargs)
        if self.algorithm == 'doc2vec':
            self.model = doc2vec.Doc2Vec(self, **kwargs)
            
        savepath = self.save_dir.joinpath(self.algorithm + '_' + arg_string)
        self.model.save(str(savepath))
        return self.model
        
    def compare_models(self, prefix=None):
        prefix = self.algorithm if prefix is None else prefix
        question_file = gensim.test.utils.datapath('questions-words.txt')
        model_accuracy_dict = {}
        for path in self.save_dir.glob(prefix + '*'):
            if self.algorithm == 'word2vec':
                wv = word2vec.Word2Vec.load(str(path)).wv
            if self.algorithm == 'doc2vec':
                wv = doc2vec.Doc2Vec.load(str(path)).wv
            acc_info = wv.accuracy(question_file, restrict_vocab=len(wv.vocab))
            tot_acc = 100 * len(acc_info[-1]['correct']) / \
                (len(acc_info[-1]['correct'] + acc_info[-1]['incorrect']))
            model_accuracy_dict[path.name] = tot_acc
            print(path.name, tot_acc)
        
        return model_accuracy_dict
        
    def to_tensorboard(model, output_path, name, labels=None):
        meta_file = name + ".tsv"
        placeholder = np.zeros((len(model.wv.index2word), model.wv.vector_size))

        with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
            file_metadata.write("Text\tFrequency\n".encode('utf-8'))
            for i, word in enumerate(model.wv.index2word):
                placeholder[i] = model[word]
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

def hyper_param_comparison():
    accuracy_dict = trainer.compare_models()
    hyper_param_dict = {}
    for key, value in accuracy_dict.items():
        for param in ['alpha', 'negative', 'sample', 'window']:
            start = key.find(param)
            end = key[start:].find('_')
            string = key[start:start+end]
            if string not in hyper_param_dict.keys():
                hyper_param_dict[string] = []
            hyper_param_dict[string].append(value) 
    for key in hyper_param_dict:
        hyper_param_dict[key] = np.mean(hyper_param_dict[key])
    print(hyper_param_dict)

if __name__ =='__main__':
    trainer = Trainer(save_dir='~/discursive_distributions/prison_corpus', 
        algorithm='word2vec')
    model = trainer.train(alpha=0.04, cbow_mean=1, min_count=10, iter=20, 
        hs=1, sample=0.001, sg=0, size=300, window=5, workers=4)
    trainer.compare_models('word2vec_alpha=0.04_cbow_mean=1_min_count=10_iter=20_hs=1_sample=0.001_sg=0_size=300_window=5_workers=4')
    hyper_param_comparison()
    
    # word vector analysis object
    wv = model.wv
    print(wv.most_similar('greed'))
    # export to tensorboard:
    to_tensorboard(model, 'corpus/tensorboard', 'count')

# to load previously trained model:
# model = doc2vec.Doc2Vec.load('corpus/doc2vec_dm=1_dbow_words=0_dm_mean=0_iter=5')

# doc2vec
# trainer = Trainer('corpus', 'doc2vec')
# model = trainer.train(alpha=0.04, cbow_mean=1, min_count=10, epochs=20, 
#     negative=8, sample=0.001, size=300, window=5, workers=4)
# hyper_param_comparison()
