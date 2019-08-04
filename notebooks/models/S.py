import nltk
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def model_dnnS1(train_data, val_data, test_data):
    X_train, y_train = train_data['statement'], train_data['label']
    X_val, y_val = val_data['statement'], val_data['label']
    X_test, y_test = test_data['statement'], test_data['label']

    vectorizer = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                 lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    X_train_onehotvector = vectorizer.fit_transform(X_train)

    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("\n\n============ Model Summary ============")
    model.summary()

    print("\n\n============ Model Training ============")
    model.fit(X_train_onehotvector, y_train,
              epochs=2, batch_size=128, verbose=1,
              validation_data=(vectorizer.transform(X_val), y_val))

    print("\n\n============ Model Evaluation ============")
    scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
    print("Accuracy:", scores[1])


def model_dnnS2(train_data, val_data, test_data):
    X_train, y_train = train_data['statement'], train_data['label']
    X_val, y_val = val_data['statement'], val_data['label']
    X_test, y_test = test_data['statement'], test_data['label']

    vectorizer = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                 lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    X_train_onehotvector = vectorizer.fit_transform(X_train)

    model = Sequential()

    model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
    model.add(Dropout(0.5))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("\n\n============ Model Summary ============")
    model.summary()

    print("\n\n============ Model Training ============")
    model.fit(X_train_onehotvector, y_train,
              epochs=2, batch_size=128, verbose=1,
              validation_data=(vectorizer.transform(X_val), y_val))

    print("\n\n============ Model Evaluation ============")
    scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
    print("Accuracy:", scores[1])


def model_lrS(train_data, val_data, test_data):
    X_train, y_train = train_data['statement'], train_data['label'].astype('int')
    X_val, y_val = val_data['statement'], val_data['label'].astype('int')
    X_test, y_test = test_data['statement'], test_data['label'].astype('int')

    vectorizer = CountVectorizer(binary=True, stop_words=nltk.corpus.stopwords.words('english'),
                                 lowercase=True, min_df=3, max_df=0.9, max_features=3000)

    X_train_onehotvector = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    print("\n\n============ Model Summary ============")
    print(clf)

    print("\n\n============ Model Training ============")
    clf = clf.fit(X_train_onehotvector, y_train)

    print("\n\n============ Model Evaluation ============")
    scores = clf.score(vectorizer.transform(X_val), y_val)
    print("Validation data Accuracy:", scores)
    scores = clf.score(vectorizer.transform(X_test), y_test)
    print("Test data Accuracy:", scores)
