Fake News Detection
===================

File structure
--------------

.

├── Fake News Detection.pdf

├── notebooks

│   ├── data

│   │   ├── clean_data.py

│   │   ├── __init__.py

│   │   ├── load_data.py

│   │   ├── test2.tsv

│   │   ├── train2.tsv

│   │   └── val2.tsv

│   ├── data_visualization.ipynb

│   ├── model_comparision.ipynb

│   └── models

│       ├── __init__.py

│       ├── SMJ.py

│       ├── SM.py

│       └── S.py

├── README.md

└── requirements.txt

How to Run
----------

1. `notebook/model_comparision.ipynb` runs all models, please refer it to reproduce the results.
2. Requirements:
   1. python-3.6
   2. jupyter notebook
   3. keras
   4. nltk
   5. pandas
   6. sklearn
   7. matplotlib

Highest Accuracy
----------------

Six way classification: 0.2646 (Logistic Regression with statements and metadata)

Binary classification : 0.6508 (Deep Neural Network with statements, metadata and justification)

Data Preprocessing
------------------

Under `data` I have made 2 .py files: `clean_data.py` and `load_data.py`.

1. `load_data.py` load data: train_data, val_data and test_data
2. `clean_data.py` prepocesses the data, except for statement, subjet, context and justification fields which are processed based on the model.
3. Best way to load data: `clean_data(load_train)`

A major part of this project has gone into data preprocessing. Following steps were done prepare data:

1. Name the columns
2. Delete irrevalant data but deleting the some columns
3. Use `nltk.stem.PorterStemmer` to stem sentences
4. Use `nltk.corpus.stopwords.words('english')` to remove stopwords
5. Replace `nan` with appropiate values:
   1. Text field was replaced `'unknown'`
   2. Number field was replaced `'unknown'`
6. Replace labels with number labels based on `binary` or not.
7. Replace speaker, speaker_title, state and party with numbers after mapping field with numbers.
8. After throughly visualizing data and trying different sentence representation, I choose `sklearn.feature_extraction.text.CountVectorizer` to get one-hot vector representation of sentences.

Models Info
-----------

Under `models` I have made 3 files: `S.py`, `SM.py` and `SMJ.py`. These are along the same lines as the referred paper. By using this design I was able to incrementally built my models.

More details on each file:
S: Classification based only on CountVectorized representation of statements.
SM: Classification based on CountVectorized representation of statements and including other metadata except justification.
SMJ: Classification based on CountVectorized representation of statements, justification and including other metadata.

All .py files contain 3 models each:

1. `keras` sequential Deep Neural Network model with input layer, whose size depends upon `sklearn.feature_extraction.text.CountVectorizer`, one hidden layer with fixed `size=500` and activation fn. `relu` and output layer with activation fn. `sigmoid`. I have used `binary_crossentropy` loss fn. though other loss fn. can also be tried. The optimizer is `adam`, though other optimizers can also be tried. No. of epocs is restricted to 2.
2. `keras` sequential Deep Neural Network model with input layer whose size depends upon `sklearn.feature_extraction.text.CountVectorizer`, two hidden layer with fixed `size=500` and activation fn. `relu`, `dropout=0.5` and output layer with activation fn. `sigmoid`. I have used `binary_crossentropy` loss fn. though other loss fn. can also be tried. The optimizer is `adam`, though other optimizers can also be tried. No. of epocs is restricted to 2.
3. `sklearn` logistic regression model with default settings, `multi_class='multinomial'` and `solver='lbfgs'`. `max_iter` is varied keeping compute time in mind.

Future work
-----------

1. Parameter tuning can be done with DNN and LR.
2. Using validation data to check training. I noticed that accuracy reduced with overfitting. We can use validation data to stop training when accuracy reduces.
3. Since data preprocessing has been done, we can very quickly add models from `sklearn` and `keras` or make our own models.

Citations
---------

1. Data preprocessing was done completely by me.
2. Libraries: `keras`, `sklearn`, `nltk`, `pandas`, `matplotlib`
3. The choice of models and input parameters has been influenced by the paper.
4. Referred this site for keras implementation https://nlpforhackers.io/keras-intro/
