import numpy as np
import randomuserlist
import random
from tensorflow import keras


def get_user_list(lst_len):
    lst = randomuserlist
    inputs_and_result = [lst.generateList(lst_len, "vk") + lst.generateList(lst_len, "ok") + lst.generateList(lst_len, "inst") + lst.generateList(lst_len, "yt") + lst.generateList(lst_len, "tt")][0]
    random.shuffle(inputs_and_result)

    return [[i[0], i[1] * 100, i[2]] for i in inputs_and_result], [i[3] for i in inputs_and_result]


model = keras.Sequential()
hidden_layer = keras.layers.Dense(5, activation = 'relu', input_shape = (3,), name = 'hidden')
model.add(hidden_layer)
output_layer = keras.layers.Dense(5, activation = 'softmax', name = 'end')
model.add(output_layer)
sgd = keras.optimizers.SGD(learning_rate = 0.001)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy')
lst = get_user_list(40)

model.fit(np.array(lst[0]), np.array(lst[1]), epochs = 200, steps_per_epoch = 200)
results = model.predict([[30, 100, 35], [40, 100, 14], [30, 100, 26], [30, 0, 48], [30, 100, 32]], steps = 1, verbose = 0)


for i in results:
    print([round(a * 100, 1) for a in i])

for layer in model.layers:
    print(layer.get_config(), layer.get_weights())