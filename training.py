# import logging 
# logging.basicConfig(format='%(message)s', level=logging.INFO)
# %reload_ext autoreload
# %autoreload 2
# from corpus_creation import *
import numpy as np 
import gensim.models.word2vec as word2vec
import gensim.models.doc2vec as doc2vec
import gensim.test.utils
import pathlib


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
            model = word2vec.Word2Vec(sentences, **kwargs)
        if self.algorithm == 'doc2vec':
            model = doc2vec.Doc2Vec(self, **kwargs)
            
        savepath = self.save_dir.joinpath(self.algorithm + '_' + arg_string)
        model.save(str(savepath))
        return model
        
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

# doc2vec
model = doc2vec.Doc2Vec.load('corpus/doc2vec_dm=1_dbow_words=0_dm_mean=0_iter=5')
trainer = Trainer('corpus', 'doc2vec')
for dbow_words in [1,0]:
    model = trainer.train(dm=1, dbow_words=dbow_words, dm_mean=0, epochs=5)
    model = trainer.train(dm=1, dbow_words=dbow_words, dm_mean=1, epochs=5)
    model = trainer.train(dm=1, dbow_words=dbow_words, dm_concat=0, epochs=5)
    model = trainer.train(dm=1, dbow_words=dbow_words, dm_concat=1, epochs=5)
    model = trainer.train(dm=0, dbow_words=dbow_words, epochs=5)
trainer.compare_models()

accuracy_dict = corpus.compare_models()
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
hyper_param_dict

model = trainer.train(alpha=0.04, cbow_mean=1, min_count=10, epochs=20, 
    negative=8, sample=0.001, size=300, window=5, workers=4)
wv = model.wv


# word2vec
trainer = Trainer('corpus', 'word2vec')
model = trainer.train(alpha=0.04, cbow_mean=1, min_count=10, iter=20, 
    hs=1, sample=0.001, sg=0, size=300, window=5, workers=4)
trainer.compare_models('word2vec_alpha=0.04_cbow_mean=1_min_count=10_iter=20_hs=1_sample=0.001_sg=0_size=300_window=5_workers=4')
model.score(['delinquent industrious'.split(' ')])
model.score(['prison is bad'.split(' ')])

    
to_tensorboard(model, 300, 'corpus/tensorboard', 'count')

alphas = [0.035, 0.04, 0.045]
min_counts = [5,10,15]
negatives = [8,9,10]
samples = [0.0005, 0.0006]
iterations = [30,35,40]
for iteration in iterations:
        model = corpus.word2vec_train(alpha=0.04, cbow_mean=1, 
            min_count=10, iter=iteration, negative=9, 
            sample=0.0065, sg=0, size=300, window=5, workers=4)

accuracy_dict = corpus.compare_models()
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
hyper_param_dict
    