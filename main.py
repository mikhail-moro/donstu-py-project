import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def result(inputs, weight, bias):
    layer_1_1 = sigmoid(float(inputs[0]) * weight[0][0] + float(inputs[1]) * weight[0][1] + float(inputs[2]) * weight[0][2] + bias[0])
    layer_1_2 = sigmoid(float(inputs[2]) * weight[1][0] + float(inputs[1]) * weight[1][1] + float(inputs[0]) * weight[1][2] + bias[1])
    return sigmoid(layer_1_1 * weight[2][0] + layer_1_2 * weight[2][1] + bias[2])


weights = [[0.6655822426188749, 0.6898156295144532, 0.847887375024601], [0.029684015036519146, 0.3258792643205719, 0.913628153398245], [-1.038314614120177, 0.24634734918286527]]
bias = [0.23857743565725315, 0.09391416510018269, 0.21121003658715193]

sex = input("Введите ваш пол м/ж: ")
if sex == "м":
    sex = 0
else:
    sex = 1
age = int(input("Введите ваш возраст: "))
time = int(input("Введите планируемое время для соцсетей: "))


res = result([age, sex, time], weights, bias)

if 0 <= res < 0.2:
    print("Скорее всего вам подойдёт ВКонтакте")
elif 0.2 <= res < 0.4:
    print("Скорее всего вам подойдёт Одноклассники")
elif 0.4 <= res < 0.6:
    print("Скорее всего вам подойдёт Инстаграмм")
elif 0.6 <= res < 0.8:
    print("Скорее всего вам подойдёт YouTube")
elif 0.8 <= res <= 1:
    print("Скорее всего вам подойдёт TikTok")

print(res)


