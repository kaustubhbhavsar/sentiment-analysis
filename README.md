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

<!-- Tools and Libraries Used -->
## Languge and Libraries Used

*   Language: Python
*   Libraries: NLTK, Keras, TensorFlow, Tokenizer, WordCloud, Re, Matplotlib, Seaborn, Numpy, Pandas.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Project Structure -->
## Project Structure
```
├── Data                           # Data files
    ├── Raw                        # Raw files (for training and testing)
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
├── data_analysis.ipynb            # Analysing the review documents
├── helper_analysis.py             # Python script for analysis
├── helper_vocab.py                # Python script for vocabulary processing/creation
├── training_final_model.ipynb     # Final model training/tuning
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Final Notes -->
## Final Notes
To run the entire project use JupyterLab or similar IDE.

Note: Notebooks use python scripts to run.

To run the python scripts:
```
$ python helper_analysis.py
$ python helper_vocab.py
```

<p align="right">(<a href="#top">back to top</a>)</p>
