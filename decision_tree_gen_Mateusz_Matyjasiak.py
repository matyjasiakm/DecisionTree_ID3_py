__author__ = "Mateusz Matyjasiak"
import math
from random import choice


# Funkcja oblicza entropie całego zbioru U
# U - zbiór uczący
def calcualte_entropy(U):
    entropy = 0
    unique_classes = get_unique_values(len(U[0]) - 1, U)
    for class_value in unique_classes:
        p = count_value_in_column(len(U[0]) - 1, class_value, U) / len(U)
        entropy += -p * math.log(p, 2)

    return entropy


# Funkcja oblicza entropie zbioru U ze względu na atrybut d
# U - zbiór uczący
# d - numer atrybutu ze względu na który ma być obliczona entropia
def attribute_entropy(d, U):
    entropy = 0
    attribute_unique_val = get_unique_values(d, U)
    for attribute_val in attribute_unique_val:
        # Zbiór, w którym wszystkie elementy mają atrybut d z wartością attribute_val
        set_with_attribute = get_set_with_specify_value_attribute(d, attribute_val, U)
        cal_entropy = calcualte_entropy(set_with_attribute)
        entropy += (len(set_with_attribute) / len(U)) * cal_entropy
    return entropy


# Funkcja zwraca zysk informacyjny dla danego atrybutu
# d - indeks atrybutu
# U - zbiór uczący
def InfGain(d, U):
    return calcualte_entropy(U) - attribute_entropy(d, U)


# Zwraca unikalne wartości atrybutu, który znajduje się pod indeksem index
# index- indeks atrybutu
# U - zbiór uczacy
def get_unique_values(index, U):
    unique_values = []
    for value in U:
        if value[index] not in unique_values:
            unique_values.append(value[index])
    return unique_values


# Funkcja zwracająca ilość wystąpień wartości danego atrybutu w zbiorze uczącym
# column_index - indeks atrybutu, w którym poszukujemy ilości wystąpień danej wartości atrybutu
# value - wartość atrybutu, którego wystąpień szukamy
def count_value_in_column(column_index, value, U):
    count = 0
    for row in U:
        if row[column_index] == value:
            count += 1
    return count


# Funkcja zwraca zbiór uczący, w którym znajdują się wszystkie pozycje z zadaną wartością atrybutu
# attribute_index - indeks rozpatrywanego atrybutu
# attribute_value - wartość atrybutu
# U - zbiór uczący.
def get_set_with_specify_value_attribute(attribute_index, attribute_value, U):
    new_set = []
    for row in U:
        if row[attribute_index] == attribute_value:
            new_set.append(row)
    return new_set


# Funkcja zwraca indeks atrybutu, którego przyrost informacyjny jest największy.
# D - zbiór nierozpatrzonych jeszcze indeksów atrybutów
# U - zbiór uczący
def get_node_with_max_info(D, U):
    results = []
    for index in D:
        results.append((index, InfGain(index, U)))
    return max(results, key=lambda x: x[1])[0]


# Funkcja obliczająca indeks Giniego i zwracająca indeks najlepszego atrybutu z D
# D - zbiór nierozpatrzonych jeszcze atrybutów
# U _ zbiór uczący
def GiniIndex(D, U):
    unique_class_value = get_unique_values(len(U[0]) - 1, U)
    index_with_gini = []
    # Dla każdego jeszcze nie użytego atrybutu
    for index_att in D:
        gini = 0
        # Dla każdej wartości rozpatrywanego atrybutu
        for att in get_unique_values(index_att, U):
            row_with_attribute = get_set_with_specify_value_attribute(index_att, att, U)
            number_of_instances_in_U = len(row_with_attribute)
            gini_temp = 1
            for class_val in unique_class_value:
                row_with_attribute2 = get_set_with_specify_value_attribute(len(U[0]) - 1, class_val, row_with_attribute)
                gini_temp -= (len(row_with_attribute2) / number_of_instances_in_U) ** 2
            gini += gini_temp * (number_of_instances_in_U / len(U))
        index_with_gini.append((index_att, gini))
    return min(index_with_gini, key=lambda x: x[1])[0]


# Funkcja ID3 generująca rekurencyjnie drzewo decyzyjne
# D - zbiór nierozpatrzonych jeszcze indeksów atrybutów
# U - zbiór uczący
# tree - budowane drzewo decyzyjne w postaci słownika.
def ID3(D, U, split_func, tree=None):
    #Sprawdznie czy zbiór zawiera tylko jedną klase
    info = check_if_class_is_the_same(U)
    if info[0]:
        return info[1][0]
    #Wybór metody podziału
    if split_func == 0:
        #InfoGain
        node_index = get_node_with_max_info(D, U)
    else:
        #Gini
        node_index = GiniIndex(D, U)
    unique_attributes = get_unique_values(node_index, U)
    #Budowa drzewa za pomocą słownika
    if tree is None:
        tree = {}
        tree[node_index] = {}

    #Dla każdego atrybutu podziel zbiór i rekurencyjnie zbuduj drzewo
    for att in unique_attributes:
        temp_U = get_set_with_specify_value_attribute(node_index, att, U)
        D.remove(node_index)
        tree[node_index][att] = ID3(D, temp_U, split_func)
        D.append(node_index)
    return tree


# Funkcja sprawdza, czy w zbiorze uczącym jest tylko jedna klasa
# U -zbiór uczący
def check_if_class_is_the_same(U):
    unique = get_unique_values(len(U[0]) - 1, U)
    return (len(unique) == 1, unique)


# Funkcja generująca funkcję klasyfikującą, uczoną na podanym zbiorze U, klasyfikującą za pomocą wygenerowanego drzewa decyzyjnego
# attribute - zbiór atrybutów, na podstawie których zostanie podjęta decyzja
# three - wygeneroawne drzewo decyzyjne.
def train_and_get_class_func_ID3(U, ordered_attributes, split_func=0):
    tree = ID3([*range(len(U[0]) - 1)], U, split_func)

    # Zwracana jest poniższa funkcja klasyfikująca.
    def decison_maker(attributes):
        exploration_tree = tree
        # Wybór indeksu atrubutu, który jest aktualnie rozpatrywany w drzewie
        attribute_index = list(exploration_tree.keys())[0]
        # Teraz kluczami drzewa są wartości wybranego atrybutu
        exploration_tree = exploration_tree[attribute_index]
        while True:
            # Pobranie wartości atrybutu z dostarczonego elementu
            att = attributes[attribute_index]
            #Jeżeli w drzewie nie ma węzła o podanej wartości atrybutu
            # to jeżeli są podane atrybuty uporządkowane w kolejności to jest wybierany atrybut na drzewie, który jest bliżej środka
            # uporządkowanego zbioru od atrybutu którego nie ma, exception_counter określa w którą strone poszukujemy atrybutu.
            exception_counter = 0
            i = 0
            while True:
                try:
                    exploration_tree = exploration_tree[att]
                    break
                except:
                    # Jeżeli atrybuty są uporządkowane określamy gdzie jest wartością trybutu, który nie wystąpił w drzewie i wybieramy kierunek
                    # poruszania się po zbiorze uporządkowanym w celu wyboru następnego atrybutu (w kierunku środka zbioru)
                    if ordered_attributes[attribute_index] is not None and exception_counter == 0:
                        i = ordered_attributes[attribute_index].index(att)
                        if i < len(ordered_attributes[attribute_index]) / 2:
                            exception_counter = 1
                        else:
                            exception_counter = -1
                    if ordered_attributes[attribute_index] is not None:
                        # Wyboru następnego atrybutu uporządkowanego(w kierunku środka zbioru)
                        i += exception_counter
                        att = ordered_attributes[attribute_index][i]
                    else:
                        # Jeżeli atrybut nieuporządkowany wybierz losowo kolejny krok
                        key = choice(list(exploration_tree.keys()))
                        att = key

            if not isinstance(exploration_tree, dict):
                break
            attribute_index = list(exploration_tree.keys())[0]
            exploration_tree = exploration_tree[attribute_index]
        #Zwracana jest klasa
        return exploration_tree

    return decison_maker


def read_set_from_file(file_path):
    file = open(file_path, "r")
    set = file.read().splitlines()
    return [line.split(",") for line in set]
