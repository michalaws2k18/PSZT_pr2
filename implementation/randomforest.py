from random import randrange
from math import log
from time import time


def splitByValue(feature, value, train_data):
    """
    Funkcja bierze zbiór obiektów "train_data" i dzieli go na dwa zbiory
    według podanego atrybutu "feature" przyrównując jego wartość do "value"
    """
    left, right = list(), list()
    for image in train_data:
        if image[feature] < value:
            left.append(image)
        else:
            right.append(image)
    return left, right


def calcAccuracy(lista_prawdziwych_klas, lista_przewidywanych_klas):
    """
    Oblicza celność czyli procent trafionych predykcji
    """
    liczba_wlasciwych_predykcji = 0.0
    for i in range(len(lista_prawdziwych_klas)):
        if lista_prawdziwych_klas[i] == lista_przewidywanych_klas[i]:
            liczba_wlasciwych_predykcji += 1
    return liczba_wlasciwych_predykcji/len(lista_prawdziwych_klas)*100.0


def createTerminal(grupa):
    """
    Spośród grupy obiektów "grupa" wybiera klasę najczęstszą i tworzy stan terminalny, liśc
    """
    wyjscia = [row[-1]for row in grupa]
    return max(set(wyjscia), key=wyjscia.count)


def getListOfUsedClasses(images_set):
    class_values = list(set([image[-1] for image in images_set]))
    return class_values


def clacOccurrenceFrequency(images_set, class_value):
    classes = [image[-1] for image in images_set]
    wynik = classes.count(class_value)
    wynik = wynik/len(images_set)
    return wynik


def calcEntropy(images_set, class_values):
    Entropy = 0
    for class_value in class_values:
        fc = clacOccurrenceFrequency(images_set, class_value)
        Entropy += fc*log(fc)
    Entropy = -Entropy
    return Entropy


def calcInf(feature_index, feature_value, images_set, groups):
    Set_Entropy = 0
    for group in groups:
        if len(group) == 0:
            continue
        class_values = getListOfUsedClasses(group)
        iloczyn = (len(group)/len(images_set))*calcEntropy(group, class_values)
        Set_Entropy += iloczyn
    return Set_Entropy


def calcInfGain(feature_index, feature_value, images_set, groups, current_Entropy):
    wynik = current_Entropy - calcInf(feature_index, feature_value, images_set, groups)
    return wynik


def chooseFeatures(images_set, n_features):
    """
    Wybiera atrybut ,jego wartość według którego podział zbioeu treningowego jest najlespzy 
    - największy indeks giniego
    Zwraca w postaci słownika
    """
    n_index, n_value, n_InfGain, n_groups = 999, 999, float('-inf'), None # deklaracja żeby było do czego porównać 

    features = list()
    while len(features) < n_features:
        index = randrange(len(images_set[0])-1)
        features.append(index)
    current_Entropy = calcEntropy(images_set, getListOfUsedClasses(images_set))
    if(len(images_set) > 255):
        pixel_values = range(0, 255)
        for feature in features:
            for pixel_value in pixel_values:
                # Dzieli zbiór obiektow na podstawie wybranych featureów
                groups = splitByValue(feature, pixel_value, images_set)
                InfGain = calcInfGain(feature, pixel_value, images_set, groups, current_Entropy)

                if InfGain > n_InfGain:
                    n_index, n_value, n_InfGain, n_groups = feature, pixel_value, InfGain, groups
    else:
        for feature in features:
            for image in images_set:
                # Dzieli zbiór obiektow na podstawie wybranych featureów
                groups = splitByValue(feature, image[feature], images_set)
                InfGain = calcInfGain(feature, image[feature], images_set, groups, current_Entropy)

                if InfGain > n_InfGain:
                    n_index, n_value, n_InfGain, n_groups = feature, image[feature], InfGain, groups
    
    # Return a dictionary
    return {'index': n_index, 'value': n_value, 'groups': n_groups}


def split(node, max_depth, min_size, n_features, depth):
    """
    Funkcja rejkurecyjna bierze poprzedni węzeł , parametry algorytmu, aktualną głebbokość drzewa
    left, right- grupy obiektów według podziału w poprzednim węźle
    jeśli jedn z nich jest pusty to stworzy węzeł terminalny z sumy
    jęsli głebokośc osiągnie wartość maksymalną to stworzy left i right węzły terminalne
    jeśli w left jest mnije obiektów niż min_size stworzy stan terminalny  w przeciwnym wypadku wywoła sie rekurencyjnie czyli kolejny podział
    podobnie dla right

    """
    left, right = node["groups"]
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = createTerminal(left+right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = createTerminal(left), createTerminal(right)
        return

    if len(left) <= min_size:
        node['left'] = createTerminal(left)
    else:
        node['left'] = chooseFeatures(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    if len(right) <= min_size:
        node['right'] = createTerminal(right)
    else:
        node['right'] = chooseFeatures(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


def buidTree(train_images, max_depth, min_size, n_features):
    # start1 = time()
    # tworzy korzeń
    # start2 = time()
    root = chooseFeatures(train_images, n_features)
    # stop2 = time()
    # Tworzy węzły dzieci
    split(root, max_depth, min_size, n_features, 1)
    # stop1 = time()
    # print(f"Tworzenie jednego drzewa: {(stop1-start1):0.3f}")
    # print(f"Wybor atrybutu i podzial dla root: {(stop2-start2):0.3f}")
    return root


def predict(node, row):
    """ 
    Predykcja klasy dla obiektu "row" 
    funkcja rekurecyjna dla każdego weżła w drzewie 
    startuje od root
    """
    if row[node["index"]] < node["value"]:
        if isinstance(node['left'],dict):
            return predict(node['left'] , row)
        else:
            return node['left']
    else:
        if isinstance(node['right'] , dict):
            return predict(node['right'], row)
        else:
            return node['right']


def getSubset(images, ratio):
    """
    wybiera losowo ze zbioru "images" liczbę obiektów, która ma udział
    "ratio"  w całym zbiorze
    """
    number_of_objects = round(len(images)*ratio)
    sample = list()
    while len(sample) < number_of_objects:
        index = randrange(len(images))
        sample.append(images[index])
    return sample


def calcPrediction(trees, row):
    """
    To jest funkchja dla całego lasu
    dla każdego drzewa wywołuje predykcję kalsy dla obiektu "row"
    a potem wybiera najczestszą predykcję i ją zwraca
    """
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def RandomForest(train_data, test_data, max_depth, min_size, sample_size, n_trees, n_features):
    """
    Funkcja lasu loswego Train_data - obiekty treningowe
    test_data -obiekty testowe
    max_depth-maksymalna głebokosć drzewa
    min_size - minimalna ilośc obiektów, dla której powstanie stan terminalny zamiast węzła
    sample_size - współczynnik ilości obiektów ze zbioru treningowego, które zostaną wzięte pod uwagę podczas budowy drzewa
    n_trees - liczba drzew decyzyjnych w lesie losowym
    n_features - liczba atrybutów brana pod uwagę podczas wyboru podziału w węźle, według wykłądu pierwiastek z ich całkowitej liczby - czyli 28

    ogólnie algorytm najpierw tworzy las losowy na podstawie zbioru treningowego
    a potem zwraca predykcje dla zbioru testowego
    """
    # start1 = time()
    trees = list()
    for i in range(n_trees):
        if(sample_size < 1.0):
            sample_image = getSubset(train_data, sample_size)
        else:
            sample_image = train_data
        tree = buidTree(sample_image, max_depth, min_size, n_features)
        trees.append(tree)
    # stop1 = time()
    # print(f"Tworzenie aktualnekj liczby drzew: {(stop1-start1):0.3f}")
    # start2 = time()
    predictions = [calcPrediction(trees,row)for row in test_data]
    # stop2 = time()
    # print(f"Predykcja zajela: {(stop2-start2):0.3f}")
    return predictions


def runRandomForest(train_data, test_data, max_depth, min_size, sample_size, n_trees, n_features):
    """
    Funkcja wywołuje las losowy i liczy celność predykcji którą potem zwraca
    """
    pred=RandomForest(train_data, test_data, max_depth, min_size, sample_size, n_trees, n_features)
    # start = time()
    actual=[row[-1]for row in test_data]
    accuracy=calcAccuracy(actual,pred)
    # stop = time()
    # print(f"Czas liczenia dokladnosci: {(stop-start):0.3f} sekund")
    return accuracy
