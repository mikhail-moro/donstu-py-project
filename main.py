from math import exp


weights = [[[-0.09233616, -0.09484169, -0.5137412 , -0.38164628,  0.52477765],
       [ 0.18182069,  0.62269163,  0.04443356, -0.08441257, -0.23126258],
       [ 0.26208717, -1.391956  , -0.18106905, -0.16377348, -0.91910046]],
           [[0.03227055, -1.0830162, -0.45987815, 0.21449643, -0.15913789],
            [0.2648304, 1.1677712, 0.79938, -0.35905856, 0.44171396],
            [-0.5499226, -0.3998966, -0.597909, -0.70432305, 0.16170573],
            [-0.02145684, 0.04332811, -0.20603317, -0.12480557, -0.17770183],
            [0.08298005, 0.4372738, -0.23406775, 0.00777137, 0.14510638]]]

bias = [[-0.60071355,  0.02919444, -0.00400464,  0.        ,  0.13409168],
        [-0.33380628,  0.2936416 ,  0.7279624 , -1.0291904 ,  0.34147626]]


sex = input("Введите ваш пол м/ж: ")
if sex == "м":
    sex = 0
else:
    sex = 1
age = int(input("Введите ваш возраст: "))
time = int(input("Введите планируемое время для соцсетей: "))


def sort_inputs(x):
    dct = {"vk": x[0],
           "ok": x[1],
           "in": x[2],
           "yt": x[3],
           "tt": x[4]}
    sorted_tuple = sorted(dct.items(), key = lambda x: x[1])
    lst = list(sorted_tuple)[::-1]
    for i in lst:
        if i[0] == "vk":
            print(f"Вам подходит ВК на {round(100 * i[1], 1)}%")
        elif i[0] == "ok":
            print(f"Вам подходят Одноклассники на {round(100 * i[1], 1)}%")
        elif i[0] == "in":
            print(f"Вам подходит Интаграмм* на {round(100 * i[1], 1)}%")
        elif i[0] == "yt":
            print(f"Вам подходит YouTube на {round(100 * i[1], 1)}%")
        elif i[0] == "tt":
            print(f"Вам подходит TikTok на {round(100 * i[1], 1)}%")


def relu(x):
    return max(0.0, x)


def softmax(x):
    total = []
    for i in x:
        y = exp(i) / (exp(x[0]) + exp(x[1]) + exp(x[2]) + exp(x[3]) + exp(x[4]))
        total.append(y)
    return total


class NeuralNetwork:
    def __init__(self, weight, bias):
        self.weight_h = weight[0]
        self.bias_h = bias[0]
        self.weight_o = weight[1]
        self.bias_o = bias[1]

    def result(self, inputs):
        h1 = relu(inputs[0] * self.weight_h[0][0] + inputs[1] * self.weight_h[1][0] + inputs[2] * self.weight_h[2][0] + self.bias_h[0])
        h2 = relu(inputs[0] * self.weight_h[0][1] + inputs[1] * self.weight_h[1][1] + inputs[2] * self.weight_h[2][1] + self.bias_h[1])
        h3 = relu(inputs[0] * self.weight_h[0][2] + inputs[1] * self.weight_h[1][2] + inputs[2] * self.weight_h[2][2] + self.bias_h[2])
        h4 = relu(inputs[0] * self.weight_h[0][3] + inputs[1] * self.weight_h[1][3] + inputs[2] * self.weight_h[2][3] + self.bias_h[3])
        h5 = relu(inputs[0] * self.weight_h[0][4] + inputs[1] * self.weight_h[1][4] + inputs[2] * self.weight_h[2][4] + self.bias_h[4])

        v1 = h1 * self.weight_o[0][0] + h2 * self.weight_o[1][0] + h3 * self.weight_o[2][0] + h4 * self.weight_o[3][0] + h5 * self.weight_o[4][0] + self.bias_o[0]
        v2 = h1 * self.weight_o[0][1] + h2 * self.weight_o[1][1] + h3 * self.weight_o[2][1] + h4 * self.weight_o[3][1] + h5 * self.weight_o[4][1] + self.bias_o[1]
        v3 = h1 * self.weight_o[0][2] + h2 * self.weight_o[1][2] + h3 * self.weight_o[2][2] + h4 * self.weight_o[3][2] + h5 * self.weight_o[4][2] + self.bias_o[2]
        v4 = h1 * self.weight_o[0][3] + h2 * self.weight_o[1][3] + h3 * self.weight_o[2][3] + h4 * self.weight_o[3][3] + h5 * self.weight_o[4][3] + self.bias_o[3]
        v5 = h1 * self.weight_o[0][4] + h2 * self.weight_o[1][4] + h3 * self.weight_o[2][4] + h4 * self.weight_o[3][4] + h5 * self.weight_o[4][4] + self.bias_o[4]

        return softmax([v1, v2, v3, v4, v5])


nn = NeuralNetwork(weights, bias)
res = nn.result([age, sex*100, time])

sort_inputs(res)