![Fig. 1](final_paper/figures/full_distribution.png)

## Code layout

Scripts 1. and 2. below do all the heavy lifting, and both can be sensibly run from the command line (but you might want to change the save directory first).

1. [corpus_creation.py](corpus_creation.py): `Corpus` class reads in PDFs/EPUBs/text files, processes them, and saves the cleaned tokens to a text file for whichever NLP model. This takes a while to run--roughly half on hour on my laptop.
2. `training.py` `Trainer` does the training and produces a Wor2Vec (or whatever) model that can be used for analysis. Also has functionality to assess and compare model accuracy, as well as porting. Doesn't take long.
3. `analysis.py` Very messy, mostly just my exploration sandbox. Has clustering at the bottom.  
4. `visualization.py` Has a single function to port a gensim word vector instance into Tensorboard.  
