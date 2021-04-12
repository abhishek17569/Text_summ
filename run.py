#pip install bert-extractive-summarizer
#pip install spacy==2.1.3
#pip install transformers==2.2.2
#pip install neuralcoref

#python -m spacy download en_core_web_md

import array
import string
import operator

from summarizer import Summarizer
import nltk
# nltk.download() --> run once, to install the ntkl packages  
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from flask import Flask, render_template, request #Used to render .html templates

#Webscrapping using BeautifulSoup, not yet implemented
from bs4 import BeautifulSoup
from urllib.request import urlopen
from typing import List, Optional, Tuple

import numpy as np
from transformers import *

from summarizer.bert_parent import BertParent
from summarizer.cluster_features import ClusterFeatures
from summarizer.sentence_handler import SentenceHandler


class ModelProcessor(object):

    aggregate_map = {
        'mean': np.mean,
        'min': np.min,
        'median': np.median,
        'max': np.max
    }

    def __init__(
        self,
        model: str = 'bert-base-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        """
        This is the parent Bert Summarizer model. New methods should implement this class

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param sentence_handler: The handler to process sentences. If want to use coreference, instantiate and pass CoreferenceHandler instance
        :param random_state: The random state to reproduce summarizations.
        """

        np.random.seed(random_state)
        self.model = BertParent(model, custom_model, custom_tokenizer)
        self.hidden = hidden
        self.reduce_option = reduce_option
        self.sentence_handler = sentence_handler
        self.random_state = random_state

    def process_content_sentences(self, body: str, min_length: int = 40, max_length: int = 600) -> List[str]:
        """
        Processes the content sentences with neural coreference.
        :param body: The raw string body to process
        :param min_length: Minimum length that the sentences must be
        :param max_length: Max length that the sentences mus fall under
        :return: Returns a list of sentences with coreference applied.
        """

        doc = self.nlp(body)._.coref_resolved
        doc = self.nlp(doc)
        return [c.string.strip() for c in doc.sents if max_length > len(c.string.strip()) > min_length]

    def cluster_runner(
        self,
        content: List[str],
        ratio: float = 0.2,
        algorithm: str = 'kmeans',
        use_first: bool = True,
        num_sentences: int = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Runs the cluster algorithm based on the hidden state. Returns both the embeddings and sentences.

        :param content: Content list of sentences.
        :param ratio: The ratio to use for clustering.
        :param algorithm: Type of algorithm to use for clustering.
        :param use_first: Whether to use first sentence (helpful for news stories, etc).
        :param num_sentences: Number of sentences to use for summarization.
        :return: A tuple of summarized sentences and embeddings
        """

        if num_sentences is not None:
            num_sentences = num_sentences if use_first else num_sentences

        hidden = self.model(content, self.hidden, self.reduce_option)
        hidden_args = ClusterFeatures(hidden, algorithm, random_state=self.random_state).cluster(ratio, num_sentences)

        if use_first:

            if not hidden_args:
                hidden_args.append(0)

            elif hidden_args[0] != 0:
                hidden_args.insert(0, 0)

        sentences = [content[j] for j in hidden_args]
        embeddings = np.asarray([hidden[j] for j in hidden_args])

        return sentences, embeddings

    def __run_clusters(
        self,
        content: List[str],
        ratio: float = 0.2,
        algorithm: str = 'kmeans',
        use_first: bool = True,
        num_sentences: int = None
    ) -> List[str]:
        """
        Runs clusters and returns sentences.

        :param content: The content of sentences.
        :param ratio: Ratio to use for for clustering.
        :param algorithm: Algorithm selection for clustering.
        :param use_first: Whether to use first sentence
        :param num_sentences: Number of sentences. Overrides ratio.
        :return: summarized sentences
        """

        sentences, _ = self.cluster_runner(content, ratio, algorithm, use_first, num_sentences)
        return sentences

    def __retrieve_summarized_embeddings(
            self, content: List[str], ratio: float=0.2, algorithm: str='kmeans', use_first: bool = True, num_sentences: int = None
    ) -> np.ndarray:
        """
        Retrieves embeddings of the summarized sentences.

        :param content: The content of sentences.
        :param ratio: Ratio to use for for clustering.
        :param algorithm: Algorithm selection for clustering.
        :param use_first: Whether to use first sentence
        :return: Summarized embeddings
        """

        _, embeddings = self.cluster_runner(content, ratio, algorithm, use_first, num_sentences)
        return embeddings

    def run_embeddings(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans',
        num_sentences: int = None,
        aggregate: str = None
    ) -> Optional[np.ndarray]:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the embeddings.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param num_sentences: Number of sentences to use. Overrides ratio.
        :param aggregate: One of mean, median, max, min. Applied on zero axis
        :return: A summary embedding
        """

        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            embeddings = self.__retrieve_summarized_embeddings(sentences, ratio, algorithm, use_first, num_sentences)

            if aggregate is not None:

                assert aggregate in ['mean', 'median', 'max', 'min'], "aggregate must be mean, min, max, or median"
                embeddings = self.aggregate_map[aggregate](embeddings, axis=0)

            return embeddings

        return None

    def run(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans',
        num_sentences: int = None
    ) -> str:
        """
        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param num_sentences: Number of sentences to use (overrides ratio).
        :return: A summary sentence
        """

        sentences = self.sentence_handler(body, min_length, max_length)

        if sentences:
            sentences = self.__run_clusters(sentences, ratio, algorithm, use_first, num_sentences)

        return ' '.join(sentences)

    def __call__(
        self,
        body: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 600,
        use_first: bool = True,
        algorithm: str = 'kmeans',
        num_sentences: int = None
    ) -> str:
        """
        (utility that wraps around the run function)

        Preprocesses the sentences, runs the clusters to find the centroids, then combines the sentences.

        :param body: The raw string body to process
        :param ratio: Ratio of sentences to use
        :param min_length: Minimum length of sentence candidates to utilize for the summary.
        :param max_length: Maximum length of sentence candidates to utilize for the summary
        :param use_first: Whether or not to use the first sentence
        :param algorithm: Which clustering algorithm to use. (kmeans, gmm)
        :param Number of sentences to use (overrides ratio).
        :return: A summary sentence
        """

        return self.run(
            body, ratio, min_length, max_length, algorithm=algorithm, use_first=use_first, num_sentences=num_sentences
        )


class Summarizer(ModelProcessor):

    def __init__(
        self,
        model: str = 'bert-large-uncased',
        custom_model: PreTrainedModel = None,
        custom_tokenizer: PreTrainedTokenizer = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):
        """
        This is the main Bert Summarizer class.

        :param model: This parameter is associated with the inherit string parameters from the transformers library.
        :param custom_model: If you have a pre-trained model, you can add the model class here.
        :param custom_tokenizer: If you have a custom tokenizer, you can add the tokenizer here.
        :param hidden: This signifies which layer of the BERT model you would like to use as embeddings.
        :param reduce_option: Given the output of the bert model, this param determines how you want to reduce results.
        :param greedyness: associated with the neuralcoref library. Determines how greedy coref should be.
        :param language: Which language to use for training.
        :param random_state: The random state to reproduce summarizations.
        """

        super(Summarizer, self).__init__(
            model, custom_model, custom_tokenizer, hidden, reduce_option, sentence_handler, random_state
        )


class TransformerSummarizer(ModelProcessor):

    MODEL_DICT = {
        'Bert': (BertModel, BertTokenizer),
        'OpenAIGPT': (OpenAIGPTModel, OpenAIGPTTokenizer),
        'GPT2': (GPT2Model, GPT2Tokenizer),
        'CTRL': (CTRLModel, CTRLTokenizer),
        'TransfoXL': (TransfoXLModel, TransfoXLTokenizer),
        'XLNet': (XLNetModel, XLNetTokenizer),
        'XLM': (XLMModel, XLMTokenizer),
        'DistilBert': (DistilBertModel, DistilBertTokenizer),
    }

    def __init__(
        self,
        transformer_type: str = 'Bert',
        transformer_model_key: str = 'bert-base-uncased',
        transformer_tokenizer_key: str = None,
        hidden: int = -2,
        reduce_option: str = 'mean',
        sentence_handler: SentenceHandler = SentenceHandler(),
        random_state: int = 12345
    ):

        try:
            self.MODEL_DICT['Roberta'] = (RobertaModel, RobertaTokenizer)
            self.MODEL_DICT['Albert'] = (AlbertModel, AlbertTokenizer)
            self.MODEL_DICT['Camembert'] = (CamembertModel, CamembertTokenizer)
        except Exception as e:
            pass  # older transformer version

        model_clz, tokenizer_clz = self.MODEL_DICT[transformer_type]
        model = model_clz.from_pretrained(transformer_model_key, output_hidden_states=True)

        tokenizer = tokenizer_clz.from_pretrained(
            transformer_tokenizer_key if transformer_tokenizer_key is not None else transformer_model_key
        )

        super().__init__(
            None, model, tokenizer, hidden, reduce_option, sentence_handler, random_state
        )

class summarize:

	def get_summary(self, input, max_sentences):
		sentences_original = sent_tokenize(input)

		#Remove all tabs, and new lines
		if (max_sentences > len(sentences_original)):
			print ("Error, number of requested sentences exceeds number of sentences inputted")
			#Should implement error schema to alert user.
		s = input.strip('\t\n')		
		#Remove punctuation, tabs, new lines, and lowercase all words, then tokenize using words and sentences 
		words_chopped = word_tokenize(s.lower())		
		sentences_chopped = sent_tokenize(s.lower())
		stop_words = set(stopwords.words("english"))
		punc = set(string.punctuation)
		filtered_words = []
		for w in words_chopped:
			if w not in stop_words and w not in punc:
				filtered_words.append(w)
		total_words = len(filtered_words)		
		word_frequency = {}
		output_sentence = []
		for w in filtered_words:
			if w in word_frequency.keys():
				word_frequency[w] += 1.0 
			else:
				word_frequency[w] = 1.0 		
		for word in word_frequency:
			word_frequency[word] = (word_frequency[word]/total_words)
		tracker = [0.0] * len(sentences_original)
		for i in range(0, len(sentences_original)):
			for j in word_frequency:
				if j in sentences_original[i]:
					tracker[i] += word_frequency[j]		
		for i in range(0, len(tracker)):
			index, value = max(enumerate(tracker), key = operator.itemgetter(1))
			if (len(output_sentence)+1 <= max_sentences) and (sentences_original[index] not in output_sentence): 
				output_sentence.append(sentences_original[index])
			if len(output_sentence) > max_sentences:
				break	
			tracker.remove(tracker[index])		
		sorted_output_sent = self.sort_sentences(sentences_original, output_sentence)
		return (sorted_output_sent)
	def sort_sentences (self, original, output):
		sorted_sent_arr = []
		sorted_output = []
		for i in range(0, len(output)):
			if(output[i] in original):
				sorted_sent_arr.append(original.index(output[i]))
		sorted_sent_arr = sorted(sorted_sent_arr)
		for i in range(0, len(sorted_sent_arr)):
			sorted_output.append(original[sorted_sent_arr[i]])
		print (sorted_sent_arr)
		return sorted_output



#------------Flask Application---------------#

app = Flask(__name__)
@app.route('/templates', methods=['POST'])
def original_text_form():
	title = "Summarizer"
	text = request.form['input_text'] #Get text from html
	max_value = sent_tokenize(text)
	num_sent = int(request.form['num_sentences']) #Get number of sentence required in summary
	sum1 = summarize()
	summary = sum1.get_summary(text, num_sent)
	return render_template("index.html", title = title, original_text = text, output_summary = summary, num_sentences = max_value)

@app.route('/analyse_url',methods=['POST'])
def fetch_and_analyze():
	url=request.form['raw_url']
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	print(fetched_text)
	max_value = sent_tokenize(fetched_text)
	num_sent = int(request.form['num_sentences']) #Get number of sentence required in summary
	sum1 = summarize()
	summary = sum1.get_summary(fetched_text, num_sent)
	return render_template('summarizeLink.html',title='Url Summarization',original_text = fetched_text, output_summary = summary, num_sentences = max_value)

@app.route('/')
def homepage():
	title = "Text Summarizer"
	return render_template("index.html", title = title)

@app.route('/summarize_link')
def linkSumaary():
	return render_template('summarizeLink.html')	

if __name__ == "__main__":
	app.debug = True
	app.run()
