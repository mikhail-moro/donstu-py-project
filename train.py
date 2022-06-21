import math
import random

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


class Layer:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feedforward(self, inputs):
        layer_1_1 = sigmoid(float(inputs[0]) * self.weight[0][0] + float(inputs[1]) * self.weight[0][1] + float(inputs[2]) * self.weight[0][2] + self.bias[0])
        layer_1_2 = sigmoid(float(inputs[2]) * self.weight[1][0] + float(inputs[1]) * self.weight[1][1] + float(inputs[0]) * self.weight[1][2] + self.bias[1])
        return sigmoid(layer_1_1 * self.weight[2][0] + layer_1_2 * self.weight[2][1] + self.bias[2])

    def loss(self, result, true_result):
        sm = 0
        n = len(result)
        for a, b in zip(result, true_result):
            sm += (b - a) ** 2

        return sm / n

    def train(self, inputs, result):
        learn_rate = 1
        epochs = 2000

        for epoch in range(epochs):
            for x, y in zip(inputs, result):
                self.weight[0][0] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*self.weight[2][0]*x[0]*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)
                self.weight[0][1] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*self.weight[2][0]*x[1]*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)
                self.weight[0][2] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*self.weight[2][0]*x[2]*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)
                self.bias[0] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*self.weight[2][0]*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)

                self.weight[1][0] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*self.weight[2][1]*x[2]*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)
                self.weight[1][1] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*self.weight[2][1]*x[1]*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)
                self.weight[1][2] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*self.weight[2][1]*x[0]*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)
                self.bias[1] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*self.weight[2][1]*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)

                self.weight[2][0] -= learn_rate * (2*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))*(-0.999999999831127*self.bias[0] - 0.999999999831127*self.weight[0][0]*x[0] - 0.999999999831127*self.weight[0][1]*x[1] - 0.999999999831127*self.weight[0][2]*x[2])/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)
                self.weight[2][1] -= learn_rate * (2*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))*(-0.999999999831127*self.bias[1] - 0.999999999831127*self.weight[1][0]*x[2] - 0.999999999831127*self.weight[1][1]*x[1] - 0.999999999831127*self.weight[1][2]*x[0])/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)
                self.bias[2] -= learn_rate * (-1.99999999966225*2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0]))*(y - 1/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1))/(2.718281828**(-self.bias[2] - self.weight[2][0]*(self.bias[0] + self.weight[0][0]*x[0] + self.weight[0][1]*x[1] + self.weight[0][2]*x[2]) - self.weight[2][1]*(self.bias[1] + self.weight[1][0]*x[2] + self.weight[1][1]*x[1] + self.weight[1][2]*x[0])) + 1)**2)

    def getValues(self):
        print("Веса ", self.weight)
        print("Смещения ", self.bias)


lst = randomuserlist
lst_len = 200

inputs_and_result = [lst.generateList(lst_len, "vk") + lst.generateList(lst_len, "ok") + lst.generateList(lst_len, "inst") + lst.generateList(lst_len, "yt") + lst.generateList(lst_len, "tt")][0]
inputs = [i[:3] for i in inputs_and_result]
result = [i[3] for i in inputs_and_result]

bias = []
weights = [[], [], []]

for i in range(len(weights)-1):
    weights[i] = [random.random(), random.random(), random.random()]
    bias.append(random.random())
weights[-1] = [random.random(), random.random()]
bias.append(random.random())


l = Layer(weights, bias)
l.train(inputs, result)
l.getValues()