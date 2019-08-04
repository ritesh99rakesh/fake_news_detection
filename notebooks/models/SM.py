import nltk
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def model_dnnSM1(train_data, val_data, test_data):
    vectorizer_statement = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                           lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    vectorizer_subject = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                         lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    vectorizer_context = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                         lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    statement_onehotvector = vectorizer_statement.fit_transform(train_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.fit_transform(train_data['subject']).toarray()
    context_onehotvector = vectorizer_context.fit_transform(train_data['context']).toarray()

    X_train, y_train = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), train_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), train_data['label'].astype('int')

    statement_onehotvector = vectorizer_statement.transform(val_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.transform(val_data['subject']).toarray()
    context_onehotvector = vectorizer_context.transform(val_data['context']).toarray()

    X_val, y_val = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), val_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), val_data['label'].astype('int')

    statement_onehotvector = vectorizer_statement.transform(test_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.transform(test_data['subject']).toarray()
    context_onehotvector = vectorizer_context.transform(test_data['context']).toarray()

    X_test, y_test = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), test_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), test_data['label'].astype('int')

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("\n\n============ Model Summary ============")
    model.summary()

    print("\n\n============ Model Training ============")
    model.fit(X_train, y_train,
              epochs=2, batch_size=128, verbose=1,
              validation_data=(X_val, y_val))

    print("\n\n============ Model Evaluation ============")
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy:", scores[1])


def model_dnnSM2(train_data, val_data, test_data):
    vectorizer_statement = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                           lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    vectorizer_subject = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                         lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    vectorizer_context = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                         lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    statement_onehotvector = vectorizer_statement.fit_transform(train_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.fit_transform(train_data['subject']).toarray()
    context_onehotvector = vectorizer_context.fit_transform(train_data['context']).toarray()

    X_train, y_train = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), train_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), train_data['label'].astype('int')

    statement_onehotvector = vectorizer_statement.transform(val_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.transform(val_data['subject']).toarray()
    context_onehotvector = vectorizer_context.transform(val_data['context']).toarray()

    X_val, y_val = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), val_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), val_data['label'].astype('int')

    statement_onehotvector = vectorizer_statement.transform(test_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.transform(test_data['subject']).toarray()
    context_onehotvector = vectorizer_context.transform(test_data['context']).toarray()

    X_test, y_test = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), test_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), test_data['label'].astype('int')

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("\n\n============ Model Summary ============")
    model.summary()

    print("\n\n============ Model Training ============")
    model.fit(X_train, y_train,
              epochs=2, batch_size=128, verbose=1,
              validation_data=(X_val, y_val))

    print("\n\n============ Model Evaluation ============")
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy:", scores[1])


def model_lrSM(train_data, val_data, test_data):
    vectorizer_statement = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                           lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    vectorizer_subject = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                         lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    vectorizer_context = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                         lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    statement_onehotvector = vectorizer_statement.fit_transform(train_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.fit_transform(train_data['subject']).toarray()
    context_onehotvector = vectorizer_context.fit_transform(train_data['context']).toarray()

    X_train, y_train = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), train_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), train_data['label'].astype('int')

    statement_onehotvector = vectorizer_statement.transform(val_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.transform(val_data['subject']).toarray()
    context_onehotvector = vectorizer_context.transform(val_data['context']).toarray()

    X_val, y_val = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), val_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), val_data['label'].astype('int')

    statement_onehotvector = vectorizer_statement.transform(test_data['statement']).toarray()
    subject_onehotvector = vectorizer_subject.transform(test_data['subject']).toarray()
    context_onehotvector = vectorizer_context.transform(test_data['context']).toarray()

    X_test, y_test = pd.concat(
        [pd.DataFrame(statement_onehotvector), pd.DataFrame(subject_onehotvector), test_data.iloc[:, 3:12],
         pd.DataFrame(context_onehotvector)], axis=1), test_data['label'].astype('int')

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=X_train.shape[1]))

    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)
    print("\n\n============ Model Summary ============")
    print(clf)

    print("\n\n============ Model Training ============")
    clf = clf.fit(X_train, y_train)

    print("\n\n============ Model Evaluation ============")
    scores = clf.score(X_val, y_val)
    print("Validation data Accuracy:", scores)
    scores = clf.score(X_test, y_test)
    print("Test data Accuracy:", scores)
