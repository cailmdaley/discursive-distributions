from glob import glob
import spacy
import textract
import gensim, gensim.test.utils
from gensim.models import word2vec, doc2vec
from gensim.models import phrases
from gensim.parsing.preprocessing import STOPWORDS
from visualization import to_tensorboard
import pathlib
import subprocess as sp

try:
    nlp('')
except:
    nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'], max_length=1e7)
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    

class Corpus:
    def __init__(self, save_dir, raw_dir):
        # get book paths, excluding spacy save files
        self.save_dir = pathlib.Path(save_dir)
        self.raw_dir = pathlib.Path(raw_dir)
        self.raw_paths = list(self.raw_dir.glob('*'))
        
    def read_book(self, path):
        # extract text as byte string, convert to unicode
        if path.name[-4:] == 'epub':
            book = open_book(path)
            lines = epub_conversion.utils.convert_epub_to_lines(book)
            extracted_sentences = []
            for line in lines:
                list_of_sentences = epub_conversion.utils.to_raw_text(line)
                for list_of_words in list_of_sentences:
                    sentence_text = ' '.join(list_of_words)
                    if len(sentence_text) > 1 and 'Google' not in list_of_words:
                        extracted_sentences.append(sentence_text)
            raw_text = '\n'.join(extracted_sentences).lower()
            raw_text = raw_text.replace('& # 39 ; s', '') # possessive
            raw_text = raw_text.replace('& # 39 ;', '')  # possessive plural
            raw_text = raw_text.replace('& # 7777 ;', '') # --
            raw_text = raw_text.replace('& quot ;', '') # quote marker
            raw_text = raw_text.replace('& lt ; !', '') # idk
            return raw_text
            
        else:
            return textract.process(path).decode().lower()
            return raw_text

    def tokenize_text(self, raw_text):
        # get rid of google header
        google_headers = [
            'see the back of the book for detailed information', 
            'https://books.google.com']
        for header in google_headers:
            index = raw_text.rfind(header)
            if index != -1:
                raw_text = raw_text[index + len(header):]
                break
            elif header == google_headers[-1]:
                raise Exception('No Google header was found in the beginning of the document.')
                
        return nlp(raw_text)
    
    def stream(self, kind, format):
        spacy_dir = self.save_dir.joinpath('spacydocs')
        spacy_dir.mkdir(exist_ok=True)
        for raw_path in self.raw_paths:
            spacy_path = spacy_dir.joinpath(raw_path.name + '.spacydoc')
            try:
                untokenized_text = spacy.tokens.Doc(spacy.vocab.Vocab()).from_disk(spacy_path)
                tokenized_text = nlp(untokenized_text.text_with_ws)
            except FileNotFoundError:
                print('Saving {} as spacydoc'.format(raw_path.name))
                raw_text = self.read_book(raw_path)
                tokenized_text = self.tokenize_text(raw_text)
                tokenized_text.to_disk(spacy_path)
                
            if kind == 'documents':
                if format == 'spacy':
                    stream_objects = tokenized_text
                if format == 'list':
                    stream_objects = [token.text for token in tokenized_text 
                    if not (token.is_punct or token.is_space)]
                yield stream_objects
                    
            elif kind == 'sentences':
                for sentence in tokenized_text.sents:
                    if format == 'spacy':
                        stream_objects = sentence
                    if format == 'list':
                        stream_objects = [token.text for token in sentence if not 
                        (token.is_punct or token.is_space)]
                    yield stream_objects
            
    def save(self, kind, bigrams=True):
        
        print('Initializing split word phraser')
        stream = self.stream('sentences', 'list')
        split_word_model = phrases.Phrases(self.stream('sentences', 'list'))
        # first, reunite words that shouldn't be split;
        # remove all bigrams that don't merge into a real word
        split_word_phraser = phrases.Phraser(split_word_model)
        for word_tuple in list(split_word_phraser.phrasegrams.keys()):
            if not word_tuple[0] + word_tuple[1] in nlp.vocab:
                del split_word_phraser.phrasegrams[word_tuple]
        # we don't want the merged words to have a delimiter in them
        split_word_phraser.delimiter = b'' 

        if bigrams is True:
            print('Initializing bigram phraser')
            # now we actually look for bigrams
            stream = self.stream('sentences', 'list')
            bigram_model = phrases.Phrases(split_word_phraser[stream])
        
            # this phraser will catch bigrams that are very unique but less
            bigram_model.min_count = 20; bigram_model.threshold=90
            bigram_phraser_threshold = phrases.Phraser(bigram_model)
        
            # this one will catch bigrams that are less unique but very common
            bigram_model.min_count = 70; bigram_model.threshold=60
            bigram_phraser_count = phrases.Phraser(bigram_model)

            
        if kind == 'documents':
            save_path = self.save_dir.joinpath('line_documents.txt')
        elif kind == 'sentences':
            sp.call(['rm -rf {}/line_sentences'.format(self.save_dir.name)], shell=True)
            save_dir = self.save_dir.joinpath('line_sentences')
            save_dir.mkdir(exist_ok=True)
        
        for i, tokenized_text in enumerate(self.stream('documents', 'spacy')):
            print('Writing {} in line-{} format'.format(self.raw_paths[i].name, kind))
            
            if kind == 'sentences':
                save_path = save_dir.joinpath(self.raw_paths[i].name +'.txt')
            if kind == 'documents':
                document_tokens = []
            with save_path.open('a') as save_file:
                for sentence in tokenized_text.sents:
                    sentence_tokens = []
                    for token in sentence:
                        if token.pos_ in ['PROPN', 'NUM']: 
                            sentence_tokens.append(token.pos_)
                        elif token.is_alpha and token.is_ascii and not token.is_oov:
                            sentence_tokens.append(token.text)
                            
                    sentence_tokens = split_word_phraser[sentence_tokens]
                    if bigrams is True:
                        sentence_tokens = bigram_phraser_threshold[sentence_tokens]
                        sentence_tokens = bigram_phraser_count[sentence_tokens]
                        
                    if kind == 'sentences':
                        sentence_string = ' '.join(sentence_tokens)
                        if len(sentence_string) > 0:
                            save_file.write(sentence_string + '\n')
                            
                    if kind == 'documents':
                        document_tokens += sentence_tokens
                if kind == 'documents':
                        document_string = ' '.join(document_tokens)
                        save_file.write(document_string + '\n')
            
if __name__ =='__main__':
    corpus = Corpus(save_dir='/Users/cdaley/discursive_distributions/prison_corpus', 
        raw_dir='prison_corpus_raw/')
    corpus.save('sentences', bigrams=True)
    
    
