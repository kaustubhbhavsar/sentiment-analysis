<!-- PROJECT NAME -->

<br />
<div align="center">
  <h3 align="center">Is this movie worth watching?</h3>
  <p align="center">
    Sentiment Analysis: a Bag-of-Words approach
    
  </p>
</div>

<!-- ABOUT PROJECT -->
## What Is It?
<a href="https://en.wikipedia.org/wiki/Sentiment_analysis#:~:text=Sentiment%20analysis%20(also%20known%20as,affective%20states%20and%20subjective%20information.">Sentiment Analysis</a> is the use of natural language processing (NLP) techniques to study the affective states and subjective information. This is widely used to summarize customer opinions and reviews for applications such as marketing, product improvement, customer service, etc. Similarly, analyzing of movie reviews is also done, to summarize the movie-goers opinion towards the movie, or rate the overall movie. These reviews are rated as positve or negative.

The aim of this project is to study various modes in bag-of-words model, and build a neural network model to predict the sentiment of the movie reviews.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- PROJECT SUMMARY -->
## Summary
The required data is provided by the Cornell University and can be downloaded directly from <a href="https://www.cs.cornell.edu/people/pabo/movie-review-data/">here</a> (polarity dataset v2.0). The dataset contains both positive and negative movie reviews. Each review is stored as a separate document in a text file.

To achieve the above mentioned aim, review documents are analyzed to understand the preprocessing steps that could help clean the documents (<a href="data_analysis.ipynb">data_analysis.ipynb</a>). A vocabulary file is generated from the training documents (<a href="helper_vocab.py">helper_vocab.py</a>). After comparing the four text encoding schemes, i.e. binary, count, tfidf, and frequency, of bag-of-words model,  it is noticed that binary encoding scheme achieves highest accuracy of 92.22% (<a href="comparative_analysis.ipynb">comparative_analysis.ipynb</a>). In the final model training, binary encoding scheme is used, and model fine tuning is performed (<a href="training_final_model.ipynb">training_final_model.ipynb</a>).

Accuracies obtained from training the network:

<div align="center">

BOW Encoding Scheme | Accuracy
:------------: | :-------------: 
binary | 0.9222
count |  0.8911
tfidf | 0.8704
freq | 0.8601
Final Model (binary + fine-tuned) | 0.8849

</div>


> NOTE: Training accuracy obtained after training final model is 0.8849 and validation accuracy is 0.8842.


> NOTE: During comparative analysis, the models may not be as robust as the final model because they were not validated or not even regularized.


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Project Structure -->
## Project Structure
```
├── Data                           # Data files
    ├── Raw                        # Raw files (zip folder)
    │   ├── neg                    # Negative review files/documents
            ├── .... 
    │   ├── pos                    # Positive review files/documents
            ├── ....
    └── Vocab                      # Vocabulary files
    │   ├── vocab_all_occ.txt      # Entire vocabulary obtained from files
    │   ├── vocab_min_occ.txt      # Vocabulary file having words with minimum occurance
├── Models                         # Saved trained models
    ├── ....                        
├── comparative_analysis.ipynb     # Comparative analysis all the bag-of-word modes
├── data_analysis.ipynb            # Analysing the review documents
├── helper_analysis.py             # Python script for analysis
├── helper_vocab.py                # Python script for vocabulary processing/creation
├── predict_.py                # Python script for predicting sentiments
├── training_final_model.ipynb     # Final model training/tuning
```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Tools and Libraries used -->
## Languge and Libraries

*   Language: Python
*   Libraries: NLTK, Keras, TensorFlow, Tokenizer, WordCloud, Re, Matplotlib, Seaborn, Numpy, Pandas.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Final Notes -->
## Final Notes
To run the entire project use JupyterLab or similar IDE.

> NOTE: Notebooks use python scripts to run.

To run the python scripts:
```
$ python helper_analysis.py
$ python helper_vocab.py
```

<p align="right">(<a href="#top">back to top</a>)</p>
