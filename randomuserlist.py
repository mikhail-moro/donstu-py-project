import random
import json


# Чтение .json файла с данными
with open('data.json', 'r') as f:
    data = json.load(f)

# Возрастные категории
age_categories = [[1, 17], [18, 24], [25, 34], [35, 44], [45, 54], [55, 71]]


# Создаёт случайно сгенеророванного пользователя выбранной соцсети
class ValuesGenerator:
    def __init__(self, soc_web):
        self.ages = [random.randint(i[0], i[1]) for i in age_categories]
        self.ages_weights = [float(i) for i in data[soc_web]["ages"].values()]
        self.time = [data[soc_web]["time"] - 2, data[soc_web]["time"] - 1, data[soc_web]["time"],
                     data[soc_web]["time"] + 1, data[soc_web]["time"] + 2]
        self.sex_weights = [list(data[soc_web]["sex"].values())[0], list(data[soc_web]["sex"].values())[1]]
        self.soc_web = soc_web
        self.total = data[soc_web]["total"]

    def getValue(self):
        age_value = random.choices(self.ages, self.ages_weights, k = 1)
        sex_value = random.choices([0, 1], self.sex_weights, k = 1)
        time_value = random.choices(self.time, k = 1)

        return [*age_value, *sex_value, *time_value, self.total]


def generateList(mass, soc_web):

    user_list = []

    for i in range(mass):
        generate = ValuesGenerator(soc_web)
        user_list.append(generate.getValue())

    return user_list