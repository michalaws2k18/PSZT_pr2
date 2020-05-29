from random import randrange


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
    Sporśród grupy obiektów "grupa" wybiera klasę najczęstszą i tworzy stan terminalny, liśc
    """
    wyjscia = [row[-1]for row in grupa]
    return max(set(wyjscia), key=wyjscia.count)


def calcEntropy(subset, labels):
    """ 
    Możńa spróbować liczyć entropię zamist indeksu giniego
    """
    pass


def calcGiniIndex(groups, class_values):
    """
    Liczy liczy indeks Giniego aby dobrać optymalną wartość podziału dla wybranego atrybutu 
    Najlepiej jak indeks Giniego jest bliski 0
    class_value - to wszystkie klasy w danym zbiorze obiektów
    Ogólnie on to robi dla każdej danej propozycji podziału
    """
    size0=float(len(groups[0]))
    size1=float(len(groups[1]))
    gini=0
    for group in groups:
        if len(group)==0:
            continue
        gini_group=0
        for class_value in class_values:
            proportion=[row[-1] for row in group].count(class_value)/float(len(group))
            gini_group+=(proportion*(1.0-proportion))
        gini+=gini_group
    return gini


def chooseFeatures(images, n_features):
    """
    Wybiera atrybut ,jego wartość według którego podział zbioeu treningowego jest najlespzy 
    - największy indeks giniego
    Zwraca w postaci słownika
    """
    b_index, b_value, b_score, b_groups = 999, 999, 999, None # deklaracja żeby było do czego porównać 
    class_values = list(set([row[-1] for row in images]))
    # Sample of all features for random forest
    features = list()
    while len(features) < n_features: 
        index = randrange(len(images[0])-1)
        features.append(index)
    for feature in features:
        for image in images:
            # Dzieli zbiór obiektow na podstawie wybranych featureów
            groups = splitByValue(feature, image[feature], images)
            gini = calcGiniIndex(groups, class_values)

            if gini < b_score:
                b_index, b_value, b_score, b_groups = feature, image[feature], gini, groups
    
    # Return a dictionary
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


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
    # tworzy korzeń
    root = chooseFeatures(train_images, n_features)
    # Tworzy węzły dzieci
    split(root, max_depth, min_size, n_features, 1)
    return root


def predict(node, row):
    """ 
    Predykcja klasy dla obiektu "row" 
    funkcja rekurecyjna dla każdego weżła w drzewie 
    startuje od root
    """
    if row[node["index"]]<node["value"]:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            return node['left']
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
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
    trees = list()
    for i in range(n_trees):
        if(sample_size<1.0):
            sample_image = getSubset(train_data, sample_size)
        tree=buidTree(sample_image,max_depth,min_size,n_features)
        trees.append(tree)
    predictions=[calcPrediction(trees,row)for row in test_data]
    return predictions


def runRandomForest(train_data, test_data, max_depth, min_size, sample_size, n_trees, n_features):
    """
    Funkcja wywołuje las losowy i liczy celność predykcji którą potem zwraca
    """
    pred=RandomForest(train_data, test_data, max_depth, min_size, sample_size, n_trees, n_features)
    actual=[row[-1]for row in test_data]
    accuracy=calcAccuracy(actual,pred)
    return accuracy