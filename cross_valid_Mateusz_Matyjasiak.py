__author__ = "Mateusz Matyjasiak"

import decision_tree_gen_Mateusz_Matyjasiak


def get_index_from_value(enumerated_list, val):
    return enumerated_list.index(val)


# Funkcja przeprowadza walidacje krzyżową i dla każdego k zwraca macierz pomyłek
# class_function_gen - funkcja która trenuje model i zwraca funkcje która klasyfikuje elementy.
# U - zbiór danych
# ordered_attributes - atrybuty których wartości są uporządkowane, jeżeli nie są to None
# split_func - metoda podziału 0 Info Gain, w.p.p Gini
# k - ilość podziałów zbioru U
def cross_validation(class_function_gen, U, ordered_attributes, split_func=0, k=3):
    # Funkcja dzieli zbiór danych U na k części
    def split_set():
        chunk_size = int(len(U) / k)
        result = []
        last_item = 0

        while last_item < len(U):
            result.append(U[last_item:last_item + chunk_size])
            last_item += chunk_size

        return result

    # Funkcja łącząca wszystkie listy z listy bez listy o indeksie index
    def concat_lists_except(index, splited_list):
        result = []
        i = -1
        for item in splited_list:
            i += 1
            if i == index:
                continue

            result = result + item
        return result

    # Podział zbioru
    splited_set = split_set()
    # Wyznaczenie wszystkich wartości klasy
    class_values = decision_tree_gen_Mateusz_Matyjasiak.get_unique_values(len(U[0]) - 1, U)
    enumerated_class_values = list(enumerate(class_values))
    all_result = []
    # Dla każdego zbioru testowego
    for i in range(k):
        test_set = splited_set[i]
        # Dane treningowe
        training_set = concat_lists_except(i, splited_set)
        # Wygenerowanie funkcji klasyfikującej, uczenie modelu
        class_function = class_function_gen(training_set, ordered_attributes, split_func)
        true_val_and_predicted = []
        # Testowanie funkcji, wynik pary (wartość oczekiwana, wartość przewidziana)
        for item in test_set:
            result = class_function(item[:-1])
            true_val_and_predicted.append((item[-1], result))
        all_result.append((i, true_val_and_predicted))

    confusion_matrices = []
    for elem in all_result:
        # Inicjacja macierzy pomyłek
        conf_matrix = [[0] * len(class_values) for _ in range(len(class_values))]
        # Wypełnienie macierzy pomyłek
        for res in elem[1]:
            index_true_class = get_index_from_value(class_values, res[0])
            index_predicted_val = get_index_from_value(class_values, res[1])
            conf_matrix[index_predicted_val][index_true_class] += 1
        confusion_matrices.append(conf_matrix)

    return (confusion_matrices, enumerated_class_values)


def print_matrix(matrix):
    for row in matrix:
        print(row)


# funkcja dla zbioru macierzy pomyłek wylicza miary ocen recall, precision i F1
def compute_metrics_for_matrices(confusion_matrices, enumerated_class_values):
    i = 1
    for matrix in confusion_matrices:
        print("K =", i)
        i += 1
        print(enumerated_class_values)
        print_matrix(matrix)
        for class_val in enumerated_class_values:
            recall = compute_recall_for_label(class_val[0], matrix)
            precision = compute_precision_for_label(class_val[0], matrix)
            f1 = compute_F1_for_label(precision, recall)

            print("Wartość klasy = ", class_val[1], "Recall =", recall, "Precision =", precision, "F1 = ", f1)
        print()
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print()


# Funkcja wylicza ocenę recall dla wartości klasy pod indeksem label_indeks
def compute_recall_for_label(label_index, matrix):
    sum = 0
    true_positive = matrix[label_index][label_index]
    for row in matrix:
        sum += row[label_index]
    if sum == 0.0:
        return None
    return true_positive / sum


# Funkcja wylicza ocenę precision dla wartości klasy label_indeks
def compute_precision_for_label(label_index, matrix):
    true_positive = matrix[label_index][label_index]
    suma = sum(matrix[label_index])
    if suma == 0:
        return None
    return true_positive / suma


# Funkcja wylicza ocenę F1 dla wartości klasy label_indeks z precision i recall
def compute_F1_for_label(precision, recall):
    if precision is None or recall is None:
        return None
    if (precision + recall) == 0:
        return None
    return 2 * (precision * recall / (precision + recall))
