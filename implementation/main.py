from mnist import MNIST
from randomforest import RandomForestwithValidation, runRandomForest, RandomForest, predictAllset
from random import seed
from math import sqrt
from time import time

if __name__ == '__main__':

    # Dane w katalogu "samples"
    start = time()
    mndata = MNIST('samples')

    # wczytanie danych
    images, labels = mndata.load_training()

    imagest, labelst = mndata.load_testing()

    """
    
    Każdy obiekt to takie jakby zdjęcie 28x28 z poziomem jasności (tylko jeden nie RGB)
    Czyli każdy obiekt ma 784 atrybuty
    Obiektów treningowych jest 60 0000
    Obiektów testowych jest 10 0000
    Zabawy z danymi , wyświetlanie itp. 
    """

    # index = random.randrange(0, len(images))  # choose an index ;-)
    # imagesNP = mndata.process_images_to_numpy(images)
    # print(mndata.display(images[2]))
    # print(type(images))
    # print(type(labels))
    # print(labels[2])
    # print("Dlugosci zbioru treningowego:")
    # print(len(labels))
    # print(len(images))
    # # print(type(imagesNP))
    # print("Dlugosci zbioru testowego:")
    # print(len(labelst))
    # print(len(imagest))
    """
    łacze klasy (cyfry - labels) z atrybutami 
    czyli teraz mam 785 atrybutów gdzie ostatnim jest klasa
    """
    
    for i in range(len(labels)):
        images[i].append(labels[i])

    for i in range(len(imagest)):
        imagest[i].append(labelst[i])
    stop = time()
    print(f"Przygotowanie danych: {(stop-start):0.3f}")

    """
    Długo się liczy więcv eybieram tylko część dla celów testowych jak działą
    """
    images = images[:5000]
    imagest = imagest[:1000]

    seed(1)
    max_depth = 50  # Maksymalna głebokość drzewa
    min_size = 100  # Minimalny rozmiar węzła to znaczy że może z niego zrobić liść( węzęł terminalny jak jest w nmim mniej niż "min_size" obiektów)
    sample_size = 0.6  # Część zbioru jaka będzie brana pod uwagę do budowy drzewa decyzyjnego
    n_features = int(sqrt(len(images[0])-1))  #liczba atrybutów jakie będą wybierane do budowy drzewa (28)
    # n_trees = [1, 5, 40] #Liczba drzew decyzyjnych w lesie losowym
    # n_trees = [1, int(sqrt(len(images)))]
    n_trees = [1, 5, 10, 25, 50]
    k_validation = 4

    for n_tree in n_trees:
        start2 = time()
        accuracy_validation, best_model = RandomForestwithValidation(images, max_depth, min_size, sample_size, n_tree, n_features, k_validation)
        predictions, accuracy = predictAllset(best_model, imagest)
        mean_accuracy = sum(accuracy_validation)/k_validation
        stop2 = time()
        print('Trees: %d' % n_tree)
        print('Accuracy: %.3f%%' % accuracy)
        print('Mean Accuracy: %.3f%%' % mean_accuracy)
        print(f"Czas wykonania algorytmu dla {n_tree} drzew wynosi {(stop2-start2):0.3f} sekund")
