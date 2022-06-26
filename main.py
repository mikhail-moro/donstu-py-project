from math import exp, log


def sigmoid(x):
    #return 1 / (1 + exp(-x))
    #return (2 / (1 + exp(-2 * x))) - 1
    return max(0.0, x)*0.1


def softmax(x):
    total = []
    for i in x:
        y = exp(i) / (exp(x[0]) + exp(x[1]) + exp(x[2]) + exp(x[3]) + exp(x[4]))
        total.append(y)
    return total


def logloss(output, result):
    total = 0
    for o, y in zip(output, result):
        total -= y * log(o)
    return total

class NeuralNetwork:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def result(self, inputs):
        h1 = sigmoid(inputs[0] * self.weight[0] + inputs[1] * self.weight[1] + inputs[2] * self.weight[2] + self.bias[0])
        h2 = sigmoid(inputs[0] * self.weight[3] + inputs[1] * self.weight[4] + inputs[2] * self.weight[5] + self.bias[1])
        h3 = sigmoid(inputs[0] * self.weight[6] + inputs[1] * self.weight[7] + inputs[2] * self.weight[8] + self.bias[2])
        h4 = sigmoid(inputs[0] * self.weight[24] + inputs[1] * self.weight[25] + inputs[2] * self.weight[26] + self.bias[8])

        v1 = sigmoid(h1 * self.weight[9] + h2 * self.weight[10] + h3 * self.weight[11] + h4 * self.weight[27] + self.bias[3])
        v2 = sigmoid(h1 * self.weight[12] + h2 * self.weight[13] + h3 * self.weight[14] + h4 * self.weight[28] + self.bias[4])
        v3 = sigmoid(h1 * self.weight[15] + h2 * self.weight[16] + h3 * self.weight[17] + h4 * self.weight[29] + self.bias[5])
        v4 = sigmoid(h1 * self.weight[18] + h2 * self.weight[19] + h3 * self.weight[20] + h4 * self.weight[30] + self.bias[6])
        v5 = sigmoid(h1 * self.weight[21] + h2 * self.weight[22] + h3 * self.weight[23] + h4 * self.weight[31] + self.bias[7])

        v = [v1, v2, v3, v4, v5]

        return softmax(v)


weights = [7.301664331320012, 0.570223061496471, 7.434506160761314, 5.488233630360416, 0.8539867662610575, 5.090958102107124, 12.819564122674779, 1.0547337203491507, 11.389471951561879, 3.010732702195124, 2.112025503666808, 3.8042183319935745, 2.3850964355471973, 2.310809535084475, 4.074887535629562, 6.9800117966623985, 4.938610240311849, 12.203170891674146, 2.514507600288209, 1.7622616922129342, 4.1466944703669695, 2.244863095664602, 1.6486849515258282, 4.161761216115949, 1.6586887692400494, 0.6602458775416515, 0.9786068433510086, 0.3463435658134911, 0.8115135223747391, 0.10717075513125174, 0.32634868493737246, 0.18209515236159823]
bias = [25.028931871373775, 18.228043658970336, 42.845561539370436, 1.8424682220548045, 1.172744319041811, 9.63450647702029, 1.5122661568328968, 1.3454908662177905, 3.509851910503541]

sex = input("Введите ваш пол м/ж: ")
if sex == "м":
    sex = 0
else:
    sex = 1
age = int(input("Введите ваш возраст: "))
time = int(input("Введите планируемое время для соцсетей: "))

nn = NeuralNetwork(weights, bias)
res = nn.result([age/100, sex/100, time/100])

print(f"Вам подходит ВК на {res[0]}%")
print(f"Вам подходят Одноклассники на {res[1]}%")
print(f"Вам подходит Интаграмм* на {res[2]}%")
print(f"Вам подходит YouTube на {res[3]}%")
print(f"Вам подходит TikTok на {res[4]}%")


