import math
import random
import numpy as np
import decimal

import randomuserlist

# x1, x2, x3 - входы age, sex(0 = male, 1 = female) и time соответственно
# h1, h2 - слои
# wx, bx - веса и смещения сответсвенно
#
# h1*w31 + h2*w32 + b4
# where: hx = x1/2 * wx1 + w1/2 * wx2 + bx -->
# --> (x1*w11 + x2*w12 + x3*w13 + b1)*w31 + (x3*w21 + x2*w22 + x1*w23 + b2)*w32 + b3


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    total = []
    for i in x:
        y = math.exp(i)/(math.exp(x[0]) + math.exp(x[1]) + math.exp(x[2]) + math.exp(x[3]) + math.exp(x[4]))
        total.append(y)
    return total


class NeuralNetwork:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feedforward(self, inputs):
        h1 = sigmoid(inputs[0] * self.weight[0] + inputs[1] * self.weight[1] + inputs[2] * self.weight[2] + self.bias[0])
        h2 = sigmoid(inputs[0] * self.weight[3] + inputs[1] * self.weight[4] + inputs[2] * self.weight[5] + self.bias[1])
        h3 = sigmoid(inputs[0] * self.weight[6] + inputs[1] * self.weight[7] + inputs[2] * self.weight[8] + self.bias[2])

        v1 = h1 * self.weight[9] + h2 * self.weight[10] + h3 * self.weight[11] + self.bias[3]
        v2 = h1 * self.weight[12] + h2 * self.weight[13] + h3 * self.weight[14] + self.bias[4]
        v3 = h1 * self.weight[15] + h2 * self.weight[16] + h3 * self.weight[17] + self.bias[5]
        v4 = h1 * self.weight[18] + h2 * self.weight[19] + h3 * self.weight[20] + self.bias[6]
        v5 = h1 * self.weight[21] + h2 * self.weight[22] + h3 * self.weight[23] + self.bias[7]

        v = [v1, v2, v3, v4, v5]

        return softmax(v)



    def loss(self, result, true_result):
        sm = 0
        n = len(result)
        for a, b in zip(result, true_result):
            sm += (b - a) ** 2

        return sm / n

    def train(self, inputs, result):
        learn_rate = 0.0001
        epochs = 500

        for epoch in range(epochs):
            for x, y in zip(inputs, result):
                self.weight[0][0] -= learn_rate * round((-1.99999999966225*2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2]))*self.weight[1][0]*self.weight[2][0]*x[0]*(y - 1/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1))/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1)**2), 300)
                self.weight[0][1] -= learn_rate * round((-1.99999999966225*2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2]))*self.weight[1][0]*self.weight[2][0]*x[1]*(y - 1/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1))/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1)**2), 300)
                self.bias[0] -= learn_rate * round((-1.99999999966225*2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2]))*self.weight[1][0]*self.weight[2][0]*(y - 1/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1))/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1)**2), 300)

                self.weight[1][0] -= learn_rate * round((1.99999999966225*2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2]))*self.weight[2][0]*(y - 1/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1))*(-self.bias[0] - self.weight[0][0]*x[0] - self.weight[0][1]*x[1])/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1)**2), 300)
                self.weight[1][1] -= learn_rate * round((-1.99999999966225*2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2]))*self.weight[2][0]*x[2]*(y - 1/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1))/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1)**2), 300)
                self.bias[1] -= learn_rate * round((-1.99999999966225*2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2]))*self.weight[2][0]*(y - 1/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1))/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1)**2), 300)

                self.weight[2][0] -= learn_rate * round((2*2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2]))*(y - 1/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1))*(-0.999999999831127*self.bias[1] - 0.999999999831127*self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - 0.999999999831127*self.weight[1][1]*x[2])/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1)**2), 300)
                self.bias[2] -= learn_rate * round((1.99999999966225*2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2]))*(y - 1/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1))/(2.718281828**(self.bias[2] + self.weight[2][0]*(-self.bias[1] - self.weight[1][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1]) - self.weight[1][1]*x[2])) + 1)**2), 300)

                if epoch % 20 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, inputs)
                    less = self.loss(result, y_preds)
                    print("Epoch %d loss: %.30f" % (epoch, less))
                    if less <= 0.08:
                        break

    def getValues(self):
        print("Веса ", self.weight)
        print("Смещения ", self.bias)


lst = randomuserlist
lst_len = 200

inputs_and_result = [lst.generateList(lst_len, "vk") + lst.generateList(lst_len, "ok") + lst.generateList(lst_len, "inst") + lst.generateList(lst_len, "yt") + lst.generateList(lst_len, "tt")][0]
random.shuffle(inputs_and_result)

inputs = [i[:3] for i in inputs_and_result]
result = [i[3] for i in inputs_and_result]

bias = []
weights = [[], [], []]

for i in range(len(weights)-1):
    weights[i] = [np.random.normal(), np.random.normal()]
    bias.append(np.random.normal())
weights[-1] = [np.random.normal()]
bias.append(np.random.normal())


#l = Layer(weights, bias)
#l.train(inputs, result)
#l.getValues()

v = [4, 5, 6]
print(softmax(v))