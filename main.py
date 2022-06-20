import math
import random


# x1, x2 - входы age и sex(0 = male, 1 = female) соответственно
# h1, h2 - слои
# wx, bx - веса и смещения сответсвенно
#
# h1*w31 + h2*w32 + b3
# where: hx = x1/2 * wx1 + w1/2 * wx2 + bx -->
# --> (x1*w11 + x2*w12 + b1)*w31 + (x2*w21 + x1*w22 + b2)*w32 + b3


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Layer:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feedforward(self, inputs):
        return sigmoid(float(inputs[0]) * self.weight[0] + float(inputs[1]) * self.weight[0] + self.bias)


age = 18
sex = 0

x = [age, sex]
b = []
w = [[], [], []]


for i in range(len(w)):
    w[i] = [random.random(), random.random()]
    b.append(random.random())


layer_1_1 = Layer(w[0], b[0])
layer_1_2 = Layer(w[1], b[1])
layer_2 = Layer(w[2], b[2])


print(layer_2.feedforward([layer_1_1.feedforward(x), layer_1_2.feedforward(x[::-1])]))
