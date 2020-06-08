from mnist import MNIST
from random import seed
from math import sqrt
from time import time

from sklearn.ensemble import RandomForestClassifier

# Dane w katalogu "samples"
mndata = MNIST('samples')

# wczytanie danych
images, labels = mndata.load_training()
imagest, labelst = mndata.load_testing()

images1 = images[:10000]
labels1 = labels[:10000]

imagest1 = imagest[:1000]
labelst1 = labelst[:1000]

def acc(prediction, labels):
    sum = 0
    for i in range(len(prediction)):
        if prediction[i] == labels[i]:
            sum += 1
    return sum/len(prediction)*100


seed(1)
# n_folds = 3 #k- walidacja testowa


#maksymalna glebokosc
max_depth = 50
#minimalna liczba próbek do splitowania node'a
min_samples_split = 100
#kryterium oceny
kryterium = 'entropy' #moze byc 'entropy' lub 'gini'
#liczba drzew
vn_trees = [1, 5, 10, 25, 50]
#ograniczenie features
max_features = 'sqrt' #moze byc 'sqrt' , None, 'log2' i 'auto' jako (default - wtedy jest sqrt) 
#
random_state = None# moze byc None albo 1


for n_trees in vn_trees:
    start = time()
    classifier = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, criterion=kryterium,random_state=random_state, max_features=max_features)
    classifier.fit(images1, labels1)
    prediction = classifier.predict(imagest1)
    accuracy = acc(prediction, labelst1)
    stop = time()
    print("n_trees: {} Trafność wyniosła: {:2.2f}% Czas: {:0.3f} sekund".format(n_trees, accuracy, stop - start))


#print(classifier.score(imagest, labelst))
