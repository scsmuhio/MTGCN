# MTGCN_Resources

### Overview
This repository contains the code and the dataset used.

### Dataset
* *Data.xlsx* file has the required data
* Different columns in the dataset are as follows : {Sentence, Sentiment, Emotion, Hatespeech, Sarcasm}
* Sentiment Labels: {"pos":Positive, "neg":Negative}
* Emotion Labels: {"happy". "sad", "fear" ,"anger"}
* Hatespeech Labels: {"yes", "no"}
* Sarcasm Labels: {"yes", "no"}
* Lexcions: Additional information on a sentence in which the particular labels  were given.

### Code
* All the required files were there in the *Code* folder.

### Word Embedding Details
<details>
<summary>Word Embeddings</summary>
	
* [BOW](#bow)
* [TF-IDF](#tf-idf)
* [Word2Vec](#word2vec)
* [GloVe](#glove)
* [FastText](#fasttext)
* [Meta-Embeddings](#meta-embeddings)
</details>

### User Interface Details
<details>
<summary>User Interface for Annotation</summary>

## How to run
* Download entire folder userinterface_annotation
* Go to /Website_with_user_login 
* "python3 app.py" command to run the file.

</details>

## Lexicon Based

## BOW

## TF-IDF

## Word2Vec
#### Code Snippet for Word2Vec Model
	import gensim
	w2vmodel = gensim.models.KeyedVectors.load_word2vec_format('./te_w2v.vec', binary=False)
* "tw_w2v.vec" file can be downloaded from "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/mounika_marreddy_research_iiit_ac_in/EYtd0as6XFZIlW-zH19YomABLvBAmrgLgc8bXv5rNOKzrw?e=4%3aRm5aaN&at=9"

## GloVe
#### Code Snippet for GloVe Model
	import gensim
	glove_model = gensim.models.KeyedVectors.load_word2vec_format('./te_glove_w2v.txt', binary=False)
* "te_glove_w2v.txt" file can be downloaded from "https://iiitaphyd-my.sharepoint.com/:t:/g/personal/mounika_marreddy_research_iiit_ac_in/EQGA3JvxTAtFpbF3CQEOI9wBDiBY6xCm3d6Q4Tk3ByZgmw?e=7JtrK1"

## FastText
#### Code Snippet for FastText Model
	import gensim
	fastText_model = gensim.models.KeyedVectors.load_word2vec_format('./te_fasttext.vec', binary=False)
* "te_fasttext.vec" file can be downloaded from "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/mounika_marreddy_research_iiit_ac_in/Ee6vQf9XLi9IroEpqaeqfbwB7_be-kS6nj69BTPhBu6LTw?e=udghNS"

## Meta-Embeddings
#### Code Snippet for Meta-Embeddings Model
	import gensim
	MetaEmbeddings_model = gensim.models.KeyedVectors.load_word2vec_format('./te_metaEmbeddings.txt', binary=False)
* "te_metaEmbeddings.txt" file can be downloaded from "https://iiitaphyd-my.sharepoint.com/:t:/g/personal/mounika_marreddy_research_iiit_ac_in/EUPGqWOW37dDn89R7wCz6_sB3TOfTX2fQkEO5PY8f2JN5A?e=IouB8R" 




