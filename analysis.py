import gensim.models.word2vec as word2vec
from gensim.parsing.preprocessing import STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from tabulate import tabulate
from sklearn import mixture, cluster, decomposition, metrics
import seaborn as sns
import pandas as pd

wv = word2vec.Word2Vec.load('corpus/word2vec_alpha=0.04_cbow_mean=1_min_count=10_iter=35_negative=9_sample=0.0065_sg=0_size=300_window=5_workers=4').wv

wv.predict_output_word([
    'of', 'the', 'commonwealth', 'to', 
    'procure', 'statements', 'of', 'the', 'affairs', 'of'])


# count vs. vector length
for i, word in enumerate(wv.index2word):
    length = np.sqrt(np.sum(wv.get_vector(word)**2) )
    count = wv.vocab[word].count
    label = labels[i] if labels[i] in labels[:5] else None
    plt.scatter(count, length, marker='.', label=labels[i], alpha=0.3)#; plt.xlim(0,10000)
plt.xscale('log'); plt.show()


[wv.index2word[index] for index in lengths.argsort()[-40:] if wv.index2word[index] not in STOPWORDS]
[wv.index2word[index] for index in lengths.argsort()[-40:] if wv.vocab[wv.index2word[index]].count < 1e3]
[wv.index2word[index] for index in lengths.argsort()[-200:] if wv.vocab[wv.index2word[index]].count > 1e3 and wv.index2word[index] not in STOPWORDS]

def relative_similarity(pos, neg=None, rows=10, topn=10**5):
    most_similar = [word[0] for word in \
        wv.most_similar_cosmul(pos, neg, topn=topn)]
    most_similar_no_neg = [word[0] for word in \
        wv.most_similar_cosmul(pos, topn=topn)]
    
    output = []
    for word in most_similar[:rows]:
        index_shift = most_similar_no_neg.index(word) - most_similar.index(word)
        if neg is not None:
            word += ' ({})'.format(index_shift)
        output.append(word)
    return output
    
def add_column(pos, neg=None, rows=10, topn=10**5):
    if neg is None:
        header = ', '.join(pos)
    else:
        header = '{} $\\to$ {} \ \n {} $\\to$ ?'.format(neg[0], pos[1], pos[0])
    column = relative_similarity(pos, neg, rows, topn)
    table[header] = column
        
table = pd.DataFrame()
n = 5
add_column(['poor', 'negro'], ['colored'], n, 1000)
add_column(['crime', 'sinner'], ['sin'], n, 1000)
add_column(['crime', 'confession'], ['sin'], n, 1000)
relative_similarity(['delinquency', 'repentance'], ['sin'], n, 10000)
print(tabulate(table, headers='keys',  tablefmt='grid', showindex=False))

table2 = table = pd.DataFrame()
add_column(['punishment'], None, 10)
add_column(['discipline'], None, 10)
add_column(['classification'], None, 10)
add_column(['colored'])
add_column(['negro'])
print(tabulate(table, headers='keys',  tablefmt='grid', showindex=False))

relative_similarity(['trial', 'priest'], ['confession'], 10, 10000)

# test = pca.transform(wv.vectors)[:1000]
test = wv.vectors[:5000]

n_clusters = 100
bgmm = mixture.BayesianGaussianMixture(n_components=n_clusters, 
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=1e7)
bgmm.fit(test); labels = bgmm.predict(test)


np.sqrt( np.mean( bgmm.covariances_**2, axis=(1,2) ) ) 



# agglom = cluster.AgglomerativeClustering(n_clusters=100, linkage='average', affinity='cosine')
# agglom.fit(wv.vectors)
# labels = agglom.labels_

# visualization.to_tensorboard(model, 'corpus/tensorboard/', 'labeled', labels)


clusters = [ [] for clust in range(n_clusters)]
for word_index, label in enumerate(labels):
    clusters[label].append(wv.index2word[word_index])
clusters = [np.array(cluster) for cluster in clusters]

for mean, cov, cluster in zip(bgmm.means_, bgmm.covariances_, clusters):
    avg_deviation =np.sqrt( np.mean( cov**2 ) )
    print(len(cluster), len(cluster)/test.shape[0]/avg_deviation**3)
    most_similar = [tup[0] for tup in wv.similar_by_vector(mean, topn=20)]
    most_frequent = [word for word in cluster if word not in STOPWORDS][:20]
    print(pd.DataFrame(data=[most_similar, most_frequent]).T)
    print('')
    
biggest_clusters = np.argsort([len(cluster) for cluster in clusters])[-5:]
sns.set_palette(sns.color_palette(n_colors=len(biggest_clusters)))
hue_dict = {}
hue_kws = {'s' : [], 'alpha' : []}
i=0
for label, clust in enumerate(clusters):
    if label in biggest_clusters: 
        hue_kws['s'].append(4)
        hue_kws['alpha'].append(1)
        hue_dict[label] = sns.color_palette()[i]
        i += 1
    else:
        hue_kws['s'].append(0.5)
        hue_kws['alpha'].append(0.2)
        hue_dict[label] = 'k'
    
    print(label, len(clust), ':', clust[:100])
wv.most_similar(['universal'])
i = 19 ; i = 31

[len(cluster) for cluster in clusters]
wv.index2entity[:100]
i = 90
i += 1; print(clusters[i][:80]); wv.similar_by_vector(bgmm.means_[i])

pca = decomposition.PCA(n_components=5); pca.fit(wv.vectors)
corner_subset = pd.DataFrame(pca.transform(wv.vectors)).iloc[:test.shape[0], :5]
corner_subset['cluster'] = labels[:test.shape[0]]
g = sns.PairGrid(corner_subset, hue='cluster', palette=hue_dict, hue_kws = hue_kws); 
g.map_lower(plt.scatter, marker='.',); g.map_diag(plt.hist); 
g.add_legend(); plt.show()
[clusters[i] for i in range(n_clusters) if i in biggest_clusters][1][:400]
