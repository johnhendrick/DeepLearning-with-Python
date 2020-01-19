from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense

embedding_layer = Embedding(1000, 64)

max_features = 1000  # number of words to consider as features
maxlen = 20
(x_train, y_train), (x_test,
                     y_test) = imdb.load_data(num_words=max_features)


x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)


model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len)
model.add(Flatten())
