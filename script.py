#!/usr/bin/env python
# coding: utf-8
# In[0]:
"""
Project of Text Analysis. Jeremy CATELAIN

INFO: All the model functions are independent. If you don't want to execute
some functions, please comment them in the main.

PEP8 style with 9.96 score.
"""
from __future__ import print_function
import glob
import os
from collections import Counter
from pprint import pprint
from time import time
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, Input, Dropout, MaxPooling1D
from keras.layers import Conv1D, Flatten
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing import text, sequence
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.feature_extraction.text import (TfidfTransformer, CountVectorizer)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier


def plot_data_analysis(ylabel, nb_words, dict_nbwords):
    """
    Plot the Analysis of the files (from the data_analysis function).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    counter = Counter(ylabel)
    ax1.barh(list(counter.keys()), list(counter.values()))
    ax1.set_title("Répartition des labels")
    counter = Counter(nb_words)
    ax2.bar(list(counter.keys()), list(counter.values()))
    ax2.set_title("Répartition des nombres de mots par documents")
    ax2.set_xlabel('Nombres de mots')
    ax2.set_ylabel('Nombres de documents')
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.figure(1)
    plt.show()
    plt.barh(list(dict_nbwords.keys()), list(dict_nbwords.values()))
    plt.title("Répartition du nombre de mots moyen par documents par label")

def data_analysis():
    """
    Analysis of the files.
    """
    df_classes = pd.read_csv("data/Tobacco3482.csv", sep=",")
    print(df_classes.shape)
    print(df_classes.columns)
    print(df_classes.describe())
    print(df_classes.sample(n=5))
    print(df_classes.dropna().shape)
    x_text = []
    y_label = []
    nb_zeros = 0
    nb_words = []
    keys = df_classes["label"]
    dict_nbwords = dict(zip(keys, [0] * len(keys)))
    for i in os.listdir("./data/Tobacco3482-OCR"):
        files = glob.glob("./data/Tobacco3482-OCR/" + i + "/*.txt")
        for file in files:
            file_txt = open(file, "r")
            text = file_txt.read()
            nb_words_text = len(Counter(text.split()))
            if nb_words_text >= 3:
                x_text.append(text)
                nb_words.append(nb_words_text)
                dict_nbwords[i] = (nb_words_text + dict_nbwords[i]) / 2
                y_label.append(i)
            else:
                nb_zeros += 1
    nb_words_array = np.array(nb_words)
    index = np.where(nb_words_array == nb_words_array.min())
    print("Nombre de documents ayant que 3 mots = ", len(index[0]))
    print("Nombre de documents ayant moins de 3 mots = ", nb_zeros)
    print("Pourcentage de documents retirés = ", np.round(100 * (nb_zeros) / 3482, 1))
    print(df(nb_words_array).describe())
    plot_data_analysis(y_label, nb_words, dict_nbwords)
    return x_text, y_label


def plot_data_creation(ytrain, ytest1, ytest, yval):
    """
    Plot informations about label for train, test and evaluation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))
    counter = Counter(ytrain)
    ax1.barh(list(counter.keys()), list(counter.values()))
    ax1.set_title("Répartition des labels train-test")
    counter = Counter(ytest1)
    ax1.barh(list(counter.keys()), list(counter.values()))
    ax1.legend(["Train", "Test"])
    counter = Counter(ytest)
    ax2.barh(list(counter.keys()), list(counter.values()))
    ax2.set_title("Répartition des labels test-val")
    counter = Counter(yval)
    ax2.barh(list(counter.keys()), list(counter.values()))
    ax2.legend(["Test", "Val"])
    plt.figure(2)
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.show()

def create_data(x_text, y_label):
    """
    create train, test and val data from the files x_text, y_label.
    """
    xtrain, xtest, ytrain, ytest1 = train_test_split(
        x_text, y_label, test_size=0.4, random_state=1
    )
    xtest, xval, ytest, yval = train_test_split(
        xtest, ytest1, test_size=0.5, random_state=1
    )
    plot_data_creation(ytrain, ytest1, ytest, yval)
    return xtrain, xtest, ytrain, ytest, xval, yval

def test_evaluation(model, xval, yval, xtest, ytest):
    """
    Evaluation of the model with evaluation and test data.
    """
    score = model.score(xval, yval)
    print("Score du modèle: ", score)
    y_pred = model.predict(xtest)
    print(classification_report(ytest, y_pred))
    matrice_confusion = confusion_matrix(ytest, y_pred)
    print("Matrice de confusion: \n", matrice_confusion)
    # normaliser la matrice de confusion
    normalization = matrice_confusion.astype(np.float).sum(axis=1)
    matrice_confusion = matrice_confusion / normalization
    matrice_confusion = np.floor(matrice_confusion * 100)
    print("Matrice de confusion normalisé (en %):", matrice_confusion.astype(int))


def model_multinomialnb(data):
    """
    Use a MultinomialNB on the data.
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]
    x_val = data[4]
    y_val = data[5]
    """
    print("\n ** MultinomialNB **")
    vectorizer = CountVectorizer()
    vectorizer.fit(data[0])
    print("Taille du vocabulaire: ", len(vectorizer.vocabulary_))
    x_train_counts = vectorizer.transform(data[0])
    x_val_counts = vectorizer.transform(data[4])
    x_test_counts = vectorizer.transform(data[1])
    print("Train: ", x_train_counts.shape)
    print("Validation: ", x_val_counts.shape)
    print("Test: ", x_test_counts.shape)
    tf_transformer = TfidfTransformer()
    # transforme la matrice en une représentation tf-idf
    x_train_tf = tf_transformer.fit_transform(x_train_counts)
    x_val_tf = tf_transformer.fit_transform(x_val_counts)
    x_test_tf = tf_transformer.fit_transform(x_test_counts)
    clf = MultinomialNB()
    clf.fit(x_train_tf, data[2])
    crossval = model_selection.cross_val_score(clf, x_val_tf, data[5], cv=10)
    print("Moyenne du score du modèle (CV): ", np.mean(crossval))
    print("Variance du score du modèle (CV): ", np.std(crossval))
    test_evaluation(clf, x_val_tf, data[5], x_test_tf, data[3])


def optimization_mnb(data):
    """
    Use a optimization parameters for the Multinomial on the data.
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]
    x_val = data[4]
    y_val = data[5]
    """
    print("\n ** Optimization MultinomialNB **")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )
    parameters = {
        "vect__max_df": (0.3, 0.4, 0.5, 0.7, 0.8),
        "vect__max_features": (1000, 1500, 2000, 2500),
        "tfidf__use_idf": (True, False),
        "vect__ngram_range": ((1, 1), (1, 2)),
        "clf__alpha": (1, 0.5, 0.1, 0.01, 0.001),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2, cv=3)
    print("Performing grid search...")
    print("Pipeline:", [name for name, _ in pipeline.steps])
    print("Paramètres:")
    pprint(parameters)
    time_0 = time()
    grid_search.fit(data[0], data[2])
    print("fait en %0.3fs" % (time() - time_0))
    print()
    print("Meilleur score: %0.3f" % grid_search.best_score_)
    print("Meilleur paramètres:")
    best_estimator = grid_search.best_estimator_
    best_parameters = best_estimator.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    crossval = model_selection.cross_val_score(best_estimator, data[4], data[5], cv=10)
    print("Moyenne du score du modèle (CV): ", np.mean(crossval))
    print("Variance du score du modèle (CV): ", np.std(crossval))
    test_evaluation(best_estimator, data[4], data[5], data[1], data[3])

    return best_parameters


def model_mlp(best_p, data):
    """
    Use a MLP on the data and using the best parameters found.
    """
    print("\n ** MLP **")
    if not bool(best_p):
        best_p["vect__max_features"] = 1000
        best_p["vect__max_df"] = 0.4
        best_p["vect__ngram_range"] = (1, 2)
    x_train = data[0]
    x_test = data[1]
    y_train = data[2]
    y_test = data[3]
    x_val = data[4]
    y_val = data[5]
    vectorizer = CountVectorizer(
        max_features=best_p["vect__max_features"],
        max_df=best_p["vect__max_df"],
        ngram_range=best_p["vect__ngram_range"],
    )
    vectorizer.fit(x_train)
    x_train_counts = vectorizer.transform(x_train)
    x_val_counts = vectorizer.transform(x_val)
    x_test_counts = vectorizer.transform(x_test)
    mlp = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(100,),
        max_iter=10,
        solver="adam",
        batch_size=40,
        verbose=1,
    )
    mlp.fit(x_train_counts, y_train)
    test_evaluation(mlp, x_val_counts, y_val, x_test_counts, y_test)


def model_logreg(best_p, data):
    """
    Use a Logistic regression on the data and use the best parameters.
    """
    print("\n ** Logistic Regression **")
    if not bool(best_p):
        best_p["vect__max_features"] = 1000
        best_p["vect__max_df"] = 0.4
        best_p["vect__ngram_range"] = (1, 2)
    xtrain = data[0]
    xtest = data[1]
    ytrain = data[2]
    ytest = data[3]
    xval = data[4]
    yval = data[5]
    vectorizer = CountVectorizer(
        max_features=best_p["vect__max_features"],
        max_df=best_p["vect__max_df"],
        ngram_range=best_p["vect__ngram_range"],
    )
    vectorizer.fit(xtrain)
    x_train_counts = vectorizer.transform(xtrain)
    x_val_counts = vectorizer.transform(xval)
    x_test_counts = vectorizer.transform(xtest)
    logr = linear_model.LogisticRegression()
    logr.fit(x_train_counts, ytrain)
    test_evaluation(logr, x_val_counts, yval, x_test_counts, ytest)


def get_train_test(train_raw_text, test_raw_text):
    """
    Tokenize and padding the data for a CNN and Embedding.
    """
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return (
        sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH),
        sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH),
    )


def get_model():
    """
    Create the model for a CNN and Embedding.
    """
    inputs = Input(shape=(MAX_TEXT_LENGTH,))
    model = Embedding(MAX_FEATURES, EMBED_SIZE, input_length=MAX_TEXT_LENGTH)(inputs)
    model = Dropout(0.5)(model)
    model = Conv1D(NUM_FILTERS, 5, padding="same", activation="relu")(model)
    model = MaxPooling1D(pool_size=30)(model)
    model = Flatten()(model)
    model = Dense(N_OUT, activation="softmax")(model)
    model = Model(inputs=inputs, outputs=model)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()
    return model

def model_cnn_load(x_text, y_label):
    """
    Use a CNN&Embedding on the data from a load model.
    """
    print("\n ** CNN Embedding (load model) **")
    x_train, x_test, y_train, y_test = train_test_split(
        x_text, y_label, test_size=0.4, random_state=1
    )
    # Convert clas string to index
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(CLASSES_LIST)
    y_train = label_enc.transform(y_train)
    y_test = label_enc.transform(y_test)
    x_vec_train, x_vec_test = get_train_test(x_train, x_test)
    print(len(x_vec_train), len(x_vec_test))
    model = load_model("modele.h5")
    model.load_weights("weights.hdf5")
    y_predicted = model.predict(x_vec_test).argmax(1)
    print("Score du modèle:", accuracy_score(y_test, y_predicted))
    # Rapport de classification
    print(classification_report(y_test, y_predicted))
    # Matrice de confusion
    matrice_confusion = confusion_matrix(y_test, y_predicted)
    # normaliser la matrice de confusion
    normalization = matrice_confusion.astype(np.float).sum(axis=1)
    matrice_confusion = matrice_confusion / normalization
    matrice_confusion = np.floor(matrice_confusion * 100)
    print("Matrice de confusion normalisé (en %):", matrice_confusion.astype(int))


def model_cnn_save(x_text, y_label):
    """
    Use a CNN&Embedding on the data with training and save model.
    """
    print("\n ** CNN Embedding (save model) **")
    x_train, x_test, y_train, y_test = train_test_split(
        x_text, y_label, test_size=0.4, random_state=1
    )
    # Convert clas string to index
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(CLASSES_LIST)
    y_train = label_enc.transform(y_train)
    y_test = label_enc.transform(y_test)
    train_y_cat = np_utils.to_categorical(y_train, N_OUT)
    # get the textual data in the correct format for NN
    x_vec_train, x_vec_test = get_train_test(x_train, x_test)
    print(len(x_vec_train), len(x_vec_test))
    model = get_model()

    callbacks = [
        ModelCheckpoint(
            "weights.hdf5",
            save_best_only=True,
            save_weights_only=True,
            monitor="val_acc",
            mode="max",
        )
    ]
    model.fit(
        x_vec_train,
        train_y_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
    )
    model.save("modele.h5")


# In[1]:
if __name__ == "__main__":
    X, Y = data_analysis()
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X_VAL, Y_VAL = create_data(X, Y)
    DATA = [X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, X_VAL, Y_VAL]
# In[2]:
    # MULTINOMIALNB
    model_multinomialnb(DATA)
# In[3]:
    BEST_PARAM = dict()
    #comment the next line if optimization is not wanted
    BEST_PARAM = optimization_mnb(DATA)
    # MLP
    model_mlp(BEST_PARAM, DATA)
    # Logistic Regression
    model_logreg(BEST_PARAM, DATA)
# In[4]:
    # Get the list of different classes
    CLASSES_LIST = np.unique(Y_TRAIN)
    N_OUT = len(CLASSES_LIST)
    # Model parameters
    MAX_FEATURES = 2000
    MAX_TEXT_LENGTH = 1213
    EMBED_SIZE = 300
    BATCH_SIZE = 16
    NUM_FILTERS = 125
    EPOCHS = 10
    VALIDATION_SPLIT = 0.1
    #CNN AND EMBEDDING
    #comment the next line if training is not wanted. Use load instead.
    model_cnn_save(X, Y)
    model_cnn_load(X, Y)
