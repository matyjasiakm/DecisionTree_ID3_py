# Klasyfikator wykorzystujący drzewa decyzyjne
Zaimplementowano algorytm tworzenia drzew decyzyjnych **ID3** oparty na 
dwóch metodach podziału: **Inforamtion Gain** oraz **indeksie
Gini’ego**. Do oceny klasyfikatora zaimplementowano k-krotną walidacje krzyżową, która dla każdego zbioru testowego wylicza z jej macierzy pomyłek, takie
miary ocen jak: precision, recall i $F_{1}$.