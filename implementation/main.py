from mnist import MNIST
import randomforest
from random import seed
from math import sqrt

if __name__ == '__main__':

    # Dane w katalogu "samples"
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
 
    """
    Długo się liczy więcv eybieram tylko część dla celów testowych jak działą
    """
    images = images[:1000]
    imagest = imagest[:100]

    """
    Tutaj łączę jkbyśmy robili z k walidacją krzyżową 
    """
    # print(len(imagest[0]))
    # for item in imagest:
    #     images.append(item)
    # print(len(images))

    seed(1)
    # n_folds = 3 #k- walidacja testowa
    max_depth = 800 #Maximum allowable depth of tree
    min_size = 1 #Minimalny rozmiar węzła to znaczy że może z niego zrobić liść( węzęł terminalny jak jest w nmim mniej niż "min_size" obiektów)
    sample_size = 0.7 #Część zbioru jaka będzie brana pod uwagę do budowy drzewa decyzyjnego
    n_features = int(sqrt(len(images[0])-1)) #liczba atrybutów jakie będą wybierane do budowy drzewa (28)
    # n_trees = [1, 5, 40] #Liczba drzew decyzyjnych w lesie losowym
    n_trees = [1, int(sqrt(len(images)))]

    for n_tree in n_trees:
        accuracy = randomforest.runRandomForest(images, imagest, max_depth, min_size, sample_size, n_tree, n_features)
        print('Trees: %d' % n_tree)
        print('Accuracy: %.3f%%' % accuracy)
        # print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))