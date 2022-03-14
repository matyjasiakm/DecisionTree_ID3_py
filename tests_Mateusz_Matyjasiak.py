__author__ = "Mateusz Matyjasiak"
from decision_tree_gen_Mateusz_Matyjasiak import read_set_from_file, train_and_get_class_func_ID3
from cross_valid_Mateusz_Matyjasiak import cross_validation, compute_metrics_for_matrices

car_set = read_set_from_file("car.data")
ordered_att = [["vhigh", "high", "med", "low"],
               ["vhigh", "high", "med", "low"],
               ["5more", "4", "3", "2"],
               ["more", "4", "2"],
               ["big", "med", "small"],
               ["high", "med", "low"]]
print("Info gain")
matrices_and_class_values = cross_validation(train_and_get_class_func_ID3, car_set, ordered_att, 0, k=5)

compute_metrics_for_matrices(matrices_and_class_values[0], matrices_and_class_values[1])

print("Gini")
matrices_and_class_values = cross_validation(train_and_get_class_func_ID3, car_set, ordered_att, 1, k=5)

compute_metrics_for_matrices(matrices_and_class_values[0], matrices_and_class_values[1])

ordered_att = [None, None, None, None, None, None, None, None, None]
tic_tac_toe_set = read_set_from_file("tic-tac-toe.data")

print("Info gain")
matrices_and_class_values = cross_validation(train_and_get_class_func_ID3, tic_tac_toe_set, ordered_att, 0, k=5)

compute_metrics_for_matrices(matrices_and_class_values[0], matrices_and_class_values[1])

print("Gini")
matrices_and_class_values = cross_validation(train_and_get_class_func_ID3, tic_tac_toe_set, ordered_att, 1, k=5)

compute_metrics_for_matrices(matrices_and_class_values[0], matrices_and_class_values[1])
