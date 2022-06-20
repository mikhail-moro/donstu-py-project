import random


class MiddleAge:
    def __init__(self, data, mass):
        self.lst = []

    def makeLst(self, x1, x2, len_koof):
        return [random.randint(x1, x2) for i in range(int(mass * len_koof))]

    def getMiddleAge(self):
        for i in data.keys():
            if i == "-18":
                self.lst += self.makeLst(0, 17, data[i])
            if i == "18-24":
                self.lst += self.makeLst(19, 24, data[i])
            if i == "25-34":
                self.lst += self.makeLst(25, 35, data[i])
            if i == "35-44":
                self.lst += self.makeLst(35, 44, data[i])
            if i == "45-54":
                self.lst += self.makeLst(45, 54, data[i])
            if i == "55+":
                self.lst += self.makeLst(55, 80, data[i])

        return self.lst


mass = 1000

data = {
    "-18": 12.1,
    "18-24": 19.2,
    "25-34": 29,
    "35-44": 21.8,
    "45-54": 9.7,
    "55+": 8.3
}

lst = []

for i in range(110):
    age = MiddleAge(data, mass)
    middleAge = age.getMiddleAge()
    sum_lst = sum(middleAge)
    len_lst = len(middleAge)
    lst.append(sum_lst / len_lst)

print(lst)

print(sum(lst) / len(lst))
