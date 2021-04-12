# Text Summarization
In the last two decades, automatic extractive text summarization on lectures has demonstrated to be a useful tool for collecting key phrases and sentences that best represent the content. However, many current approaches utilize dated approaches, producing sub-par outputs or requiring several hours of manual tuning to produce meaningful results. Recently, new machine learning architectures have provided mechanisms for extractive summarization through the clustering of output embeddings from deep learning models. This paper reports on the project called Lecture Summarization Service, a python based RESTful service that utilizes the BERT model for text embeddings and KMeans clustering to identify sentences closes to the centroid for summary selection. The purpose of the service was to provide students a utility that could summarize lecture content, based on their desired number of sentences. On top of the summary work, the service also includes lecture and summary management, storing content on the cloud which can be used for collaboration. While the results of utilizing BERT for extractive summarization were promising, there were still areas where the model struggled, providing feature research opportunities for further improvement. 

Setting up the Python environment

---
First, install the following packages.

```bash
nltk.download() --> run once, to install the ntkl packages  
```

Next, install the following Pip packages.

```bash
pip install flask
pip install -r requirements.txt
pip install bert-extractive-summarizer
pip install spacy==2.1.3
pip install transformers==2.2.2
pip install neuralcoref
python -m spacy download en_core_web_md
 
 ```

Start the application,
```bash
python run.py
```
Open "http://127.0.0.1:5000/" in browser.

