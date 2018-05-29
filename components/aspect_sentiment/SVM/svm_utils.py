def train_model(model_type = 'svc', print_results = True, vectorizer=None, train_data=None, train_labels=None, test_data=None, test_labels=None):
    import sys
    import os
    import time
    from sklearn import svm
    from sklearn.metrics import classification_report
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    import pickle
    # Initial setup, loading vectorizer and train/test data if none is provided
    if vectorizer:
        if type(vectorizer) == type(TfidfVectorizer()):
            vectorizer = vectorizer
        else:
            vectorizer = pickle.load(open(vectorizer, "rb"))
    else:
        print("No vectorizer specified, loading default")
        vectorizer = pickle.load(open("vectorizer.pickle", "rb"))

    if train_data:
        train_data = pickle.load(open(train_data, "rb"))
    else:
        print("Loading training data")
        train_data = pickle.load(open("train_data.pickle", "rb"))

    if test_data:
        test_data = pickle.load(open(test_data, "rb"))
    else:
        print("Loading test data")
        test_data = pickle.load(open("test_data.pickle", "rb"))

    if train_labels:
        #SVR models require float values as labels
        #SVC models require str values as labels
        train_labels = pickle.load(open(train_labels, "rb"))
    else:
        print("Loading training labels")
        if model_type.lower() == 'svr':
            train_labels = pickle.load(open("train_labels_SVR.pickle", "rb"))
        else:
            train_labels = pickle.load(open("train_labels.pickle", "rb"))

    if test_labels:
        test_labels = pickle.load(open(test_labels, "rb"))
    else:
        print("Loading test labels")
        if model_type.lower() == 'svr':
            test_labels = pickle.load(open("test_labels_SVR.pickle", "rb"))
        else:
            test_labels = pickle.load(open("test_labels.pickle", "rb"))

    # Convert input text to a matrix of float64 for SVM models
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    if model_type.lower() == 'svc':
        #Train an SVM classification model
        model = svm.LinearSVC()
        t0 = time.time()
        model.fit(train_vectors, train_labels)
        t1 = time.time()
        prediction_model = model.predict(test_vectors)
        t2 = time.time()
        time_model_train = t1-t0
        time_model_predict = t2-t1
        if print_results:
            print("Results for LinearSVC()")
            print("Training time: %fs; Prediction time: %fs" % (time_model_train, time_model_predict))
            print(classification_report(test_labels, prediction_model))

    elif model_type.lower() == 'svr':
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score
        #Train an SVR regression model
        model = svm.LinearSVR()
        t0 = time.time()
        model.fit(train_vectors, train_labels)
        t1 = time.time()
        prediction_model= model.predict(test_vectors)
        t2 = time.time()
        time_model_train = t1-t0
        time_model_predict = t2-t1
        if print_results:
            print("Results for LinearSVR()")
            print("Training time: %fs; Prediction time: %fs" % (time_model_train, time_model_predict))
            print("Mean squared error: ", mean_squared_error(test_labels, prediction_model))
            print("r2 score: ", r2_score(test_labels, prediction_model))
    else:
        print("Unknown model_type specified. Accepted values: 'svc', 'svc'.")
        model = None

    return(model)

def predict_sentiment(model, text, vectorizer):
    return(model.predict(vectorizer.transform([text])))

def create_vectorizer(min_df=3, max_df = 1.0, sublinear_tf=True, use_idf=True):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df=min_df,
                             max_df = max_df,
                             sublinear_tf=sublinear_tf,
                             use_idf=use_idf)
    return(vectorizer)
