from random import randint, randrange, random


class Gardes:
    def __init__(self):
        self.œuvres =\
            list(sorted(set([round(randint(1, 1000) * (1.5 - random()) ** 2) for n in range(randint(200, 500))])))
        self.gardes = []

    def placement(self):
        for index, _ in enumerate(self.œuvres):
            try:
                if self.gardes[-1] + 5 >= self.œuvres[index]:
                    continue
            except IndexError:
                pass
            self.gardes.append(self.œuvres[index] + 5)
