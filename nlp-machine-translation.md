# Notes on Machine Translation
## Table of Contents
- [Neural MT](#neural-mt)
- [Pre-trained language models](#pre-trained-language-models)
- [Text pre-processing](#text-pre-processing)
  * [Tokenization](#tokenization)
- [MT QE](#mt-qe)
  * [Metrics](#metrics)
  * [Quality estimation](#quality-estimation)
    

## Neural MT
1. seq2seq: End-to-end encoder-decoder neural network. Both encoder and decoder uses deep LSTM.
   1. [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (NIPS 2014)
1. seq2seq with attention: In original seq2seq, the LSTM-based encoder compress all the input words into a single vector and feed it to the decoder. Using attention mechanism connects the encoder outputs of each input word to decoder, which improves the translation of longer sentences.
   1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (ICLR 2015)
1. Google's NMT: How Google build a production NMT service
   1. [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)
   1. [Google’s Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://www.aclweb.org/anthology/Q17-1024/) (TACL 2017)
1. Transformer: Use multi-head self-attention in encoder and decoder to replace RNN/CNN. Transformer is more parallelizable and thus requires less training time. 
   1. [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) (NIPS 2017)
1. Amazon's NMT
   1. [The Sockeye Neural Machine Translation Toolkit at AMTA 2018](https://www.aclweb.org/anthology/W18-1820/) (AMTA 2018) Open source NMT with Gluon API of MXNet
   1. [Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation](https://www.aclweb.org/anthology/N18-1119/) (NAACL 2018) custom terminology
   1. [Training Neural Machine Translation to Apply Terminology Constraints](https://www.aclweb.org/anthology/P19-1294/) (ACL 2019) custom terminology

## Pre-trained language models 
1. word2vec: Use continuous Bag-of-Words (CBOW) model and continuous Skip-gram model to learn vector representations for words
   1. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) ([Rejected by ICLR 2013](https://openreview.net/forum?id=idpCdOWtqXd60))
   1. [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (NIPS 2013) Improve Skip-gram model with hierarchical softmax, negative sampling, and subsampling of frequent words
   1. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) (TACL 2017) [FastText](https://fasttext.cc/)
   1. [Learning Word Vectors for 157 Languages](https://arxiv.org/abs/1802.06893) (LREC 2018) [FastText multi-lingual word vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
   1. [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087) (ICLR 2018) Unsupervised approach with adversarial training to learn a linear mapping between mono-lingual word embeddings. Using cross-domain similarity local scaling (CSLS) increases the accuracy for word translation retrieval.
   1. [word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) tutorial
1. GloVe: Use global word-word co-occurrence counts in the formulation of the probability of a word appearing in a context. More efficient and higher performance than word2vec.
   1. [Glove: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162/) (EMNLP 2014)
1. Semi-Supervised Sequence Learning: Pre-trained LSTM RNN for sequence representation. Two approaches: (1) Predict what comes next in a sequence, and (2) Sequence auto-encoder.
   1. [Semi-Supervised Sequence Learning](https://papers.nips.cc/paper/5949-semi-supervised-sequence-learning) (NIPS 2015) 
1. ELMo (Embeddings from Language Models): Pre-trained word embeddings from bi-LSTM neural language model
   1. [Deep Contextualized Word Representations](https://www.aclweb.org/anthology/N18-1202/) (NAACL 2018)
1. ULMFit: An effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model
   1. [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) (ACL 2018)
1. OpenAI GPT: Uni-directional Transformer encoder representation trained with standard language modeling objective (next word prediction)
   1. [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
1. BERT: Bi-directional Transformer encoder representation trained with masked language modeling and next sentence prediction
   1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (NAACL 2019)
   1. [Multi-lingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md)
1. RoBERTa: Demonstrated that training BERT for more epochs and/or on more data improves model performance. Suggested that the original BERT is significantly under-trained.
   1. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) ([Rejected by ICLR 2020](https://openreview.net/forum?id=SyxS0T4tvS))
1. XLNet: A generalized autoregressive pretraining method with permutation language modeling objective. Incorporating Ideas from [Transformer-XL](https://arxiv.org/abs/1901.02860)
   1. [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) (NeurIPS 2019)
1. ALBERT: Using two parameter-reduction techniques (factorized embedding and cross-layer parameter sharing) to lower memory consumption and increase the training speed of BERT.
   1. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) (ICLR 2020)
1. T5: Ablation study for many aspects of pre-training
   1. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
1. ELECTRA: A more sample-efficient pre-training task called replaced token detection. Train a discriminative model that predicts whether each token in the corrupted input was replaced by a generator sample or not. Performs comparably to RoBERTa and XLNet while using less than 1/4 of their compute and outperforms them when using the same amount of compute.
   1. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) (ICLR 2020)
1. XLM: Cross-lingual Language Model Pretraining
   1. [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) (NeurIPS 2019) 
   1. [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) (ACL 2020) XLM-RoBERTa
   1. [PyTorch original implementation of Cross-lingual Language Model Pretraining](https://github.com/facebookresearch/XLM)

## Text pre-processing 
### Tokenization
   1. [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf) (ICASSP 2012)
   1. [BPE](https://www.aclweb.org/anthology/P16-1162/) (ACL 2016)
   1. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959) (ACL 2018)
   1. [SentencePiece](https://github.com/google/sentencepiece)

## MT QE 
### Metrics
   1. [Bleu: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040/) (ACL 2002) Bilingual evaluation understudy. Computed based on n-gram precision (how many of the n-grams in MT are present in the reference translation) and brevity penalty.
   1. [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://www.aclweb.org/anthology/W05-0909/) (ACL 2005)
   1. [A Study of Translation Edit Rate with Targeted Human Annotation](https://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf) (AMTA 2006) Human-targeted TER
   1. [chrF: character n-gram F-score for automatic MT evaluation](https://www.aclweb.org/anthology/W15-3049/) (WMT 2015)
### Quality estimation
   1. [Multi-level Translation Quality Prediction with QUEST++](https://www.aclweb.org/anthology/P15-4020/) (ACL-IJCNLP 2015)
   1. [Alibaba Submission for WMT18 Quality Estimation Task](https://www.aclweb.org/anthology/W18-6465.pdf) (WMT 2018 QE Task)
   1. [“Bilingual Expert” Can Find Translation Errors](https://arxiv.org/pdf/1807.09433.pdf) (AAAI 2019) Alibaba
   1. [SOURCE: SOURce-Conditional Elmo-style Model for Machine Translation Quality Estimation](http://www.statmt.org/wmt19/pdf/54/WMT11.pdf) (WMT 2019 QE Task)
   1. [Practical Perspectives on Quality Estimation for Machine Translation](https://arxiv.org/abs/2005.03519)


