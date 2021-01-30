import numpy as np
from sklearn import metrics, model_selection, svm, preprocessing
import pickle


def main():
    data = np.load("data.npy", allow_pickle=True)
    features = data[0].tolist()
    labels = data[1].tolist()
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2)
    print(len(labels), "splited to", len(X_train), "train and", len(X_test), "test")

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    
    score = model_selection.cross_val_score(clf, X_train, y_train, cv=5)
    print(score, '\nMean:', np.mean(score))

    print("\nHyper params tuning")
    param_grid = {'C': [0.01, 0.1, 0.5, 1, 5, 10, 15, 25, 50, 100],  
              'gamma': [1, 0.5, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
              }  
    grid = model_selection.GridSearchCV(svm.SVC(), param_grid, refit = True, cv=5) 
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.best_estimator_) 
    fine_tuned_clf = grid.best_estimator_

    fine_tuned_clf.fit(X_train, y_train)
    score = model_selection.cross_val_score(fine_tuned_clf, X_train, y_train, cv=5)
    print(score, '\nMean:', np.mean(score))

    default_clf = svm.SVC()
    default_clf.fit(X_train, y_train)
    y_pred = default_clf.predict(X_test)
    test_score = metrics.accuracy_score(y_pred, y_test)
    print(test_score)

    pickle.dump(default_clf, open("mmg_model.pkl", 'wb'))


if __name__ == "__main__":
    main()


# Pierwsza czynnoscia byla standaryzacja danych, ktora pozwolila uniknac zdominowania atrybutow o mniejszych wartosciach liczbowych przez te o wartosciach wiekszych. Polega ona na przeskalowaniu wszystkich probek tak, aby ich wartosc miescila sie pomiedzy 0 a 1. Uzyto do tego estymatora MinMaxScaler. Operacja zostala przedstawiona na listingu \ref{lst:standarization}

# \begin{lstlisting}[language=Python, caption=Standaryzacja danych., basicstyle=\tiny, label=lst:standarization]
# print("\nBefore standarization:\n", features[0])
# scaler = preprocessing.MinMaxScaler()
# scaler.fit(features)
# features = scaler.transform(features)
# print("\nAfter standarization:\n", features[0])
# \end{lstlisting}

# Na listingu \ref{lst:standarization_efect} znajduje sie efekt powyzszego kodu, na ktorym widac porownanie przykladowej cechy przed i po standaryzacji. 

# \begin{lstlisting}[language=Python, caption=Porownanie cech przed i po standaryzacji., basicstyle=\tiny, label=lst:standarization_efect]
# Before standarization:
#  [0.01622089 0.00361105 0.00287001 0.00326861 0.00287796 0.0046715
#  0.00202101 0.00869081]

# After standarization:
#  [0.11306727 0.02649908 0.01993073 0.03560453 0.03174739 0.0274772
#  0.01710949 0.08213378]
# \end{lstlisting}
