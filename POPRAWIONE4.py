import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Wczytanie danych z pliku CSV
df = pd.read_csv('feature_vectors.csv')

# Wyodrębnienie cech (X) i etykiet kategorii (y)
X = df.iloc[:, 2:].values  # pomijamy kolumny Category i File
y = df.iloc[:, 0].values  # kolumna Category

# Wstępne przetwarzanie danych
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse=False)
y_int = y_int.reshape(len(y_int), 1)
y_onehot = onehot_encoder.fit_transform(y_int)

# Podział zbioru na część treningową i testową
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

# Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Dense(units=10, activation='sigmoid', input_dim=X_train.shape[1]))
model.add(Dense(units=y_onehot.shape[1], activation='softmax'))  # liczba jednostek odpowiada liczbie unikalnych kategorii

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Uczenie sieci
model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

# Testowanie sieci
y_pred = model.predict(X_test)

# Przekonwertowanie wektorów y_test oraz y_pred do kodowania całkowitego
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)

# Wyliczenie macierzy pomyłek
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
