from typing import Union, Tuple, List, Optional, Generator, Iterable
from collections import deque


class AlgorithmsAssignment2:

    @classmethod
    def split(cls, s: str, t: str) -> Union[bool, Tuple[List[str], List[str]]]:
        """
        Determines if two strings come from the composition of two root strings.
        :param s: The first possible composition.
        :param t: The second possible composition.
        :return: False if the two strings are not a composition, two list of strings representing the roots otherwise.
        """
        a = sorted(s)
        b = sorted(t)
        if a != b:
            return False
        s = list(s)
        t = list(t)
        solutions = [[] for _ in range(len(s))]
        for index, solution in enumerate(solutions[:]):
            print(index)
            if index != 0:
                solution = solutions[index - 1][:]
                print(f'index: {index}')
            char = s[index]
            solution.append(char)
            if not cls._contains(''.join(t), solution):
                solution.pop()
            solutions[index] = solution
            print(solutions)
        x = solutions[-1]
        y = s.copy()
        for char in x:
            y.remove(char)
        c = sorted(x + y)
        if c != a or c != b:
            return False
        return x, y

    @staticmethod
    def _contains(a: Iterable[str], b: Iterable[str]) -> bool:
        b = deque(b)
        for char in a:
            if not b:
                return True
            if b[0] == char:
                b.popleft()
        return not b

    stages = 0

    @classmethod
    def select_slice(cls, a: int, b: int, l: List[int]) -> Tuple[int, int]:
        start = len(l) // 2
        end = len(l) // 2
        start = cls.find_start(0, len(l) - 1, a, start, l)
        end = cls.find_end(0, len(l) - 1, b, end, l)
        return start, end

    @classmethod
    def find_start(cls, low: int, high: int, a: int, start: int, l: List[int]) -> int:
        print(f'{low}, {high}, {start}')
        cls.stages += 1
        if l[start - 1] < a < l[start + 1]:
            return start
        if l[start] == a:
            try:
                while l[start] == a:
                    start -= 1
                start += 1
            except IndexError:
                start += 1
            return start
        if l[start] > a:
            new_start = start - (start - low) // 2
            return cls.find_start(low, start, a, new_start, l)
        if l[start] < a:
            new_start = start + (high - start) // 2
            return cls.find_start(start, high, a, new_start, l)

    @classmethod
    def find_end(cls, low: int, high: int, b: int, end: int, l: List[int]) -> int:
        print(f'{low}, {high}, {end}')
        cls.stages += 1
        if l[end - 1] < b < l[end + 1]:
            return end
        if l[end] == b:
            try:
                while l[end] == b:
                    end += 1
                end -= 1
            except IndexError:
                end -= 1
            return end
        if l[end] > b:
            new_end = end - (end - low) // 2
            return cls.find_end(low, end, a, new_end, l)
        if l[end] < b:
            new_end = end + (high - end) // 2
            return cls.find_end(end, high, b, new_end, l)

    @staticmethod
    def majority_naive(l):
        elements = {}
        for num in l:
            for key in elements.keys():
                if num == key:
                    elements[key].append(num)
                    break
            else:
                elements[num] = [num]
        for key, values in elements.items():
            if len(values) >= len(l) // 2:
                return key
        return False

    @classmethod
    def majority_divide(cls, l: List[int]) -> Union[bool, int]:
        return cls._majority_divide(l) or False

    @classmethod
    def _majority_divide(cls, l: List[int]) -> Optional[int]:
        if len(l) == 2:
            if l[0] == l[1]:
                return l[0]
            else:
                return
        else:
            length = len(l) // 2
            a = cls._majority_divide(l[length:])
            b = cls._majority_divide(l[:length])
            if not a and not b:
                return
            if not a:
                return b
            if not b:
                return a
            if a != b:
                return
            return a

    @classmethod
    def alignment(cls, a: str, b: str) -> List[List[int]]:
        a = [None] + list(a)
        b = [None] + list(b)
        table = [list(range(len(a)))]
        for index_b, val_b in enumerate(b[1:]):
            index_b += 1
            row = []
            table.append(row)
            for index_a, val_a in enumerate(a):
                if index_a == 0:
                    row.append(index_b)
                    continue
                delta = 0 if val_a == val_b else 1
                row.append(min(
                    table[index_b][index_a - 1] + 1,
                    table[index_b - 1][index_a] + 1,
                    table[index_b - 1][index_a - 1] + delta
                ))
        return table

    @classmethod
    def count_alignments(cls, a, b):
        table_alignments = cls.alignment(a, b)
        table = [[0 for _ in range(len(a) + 1)] for _ in range(len(b) + 1)]
        index_a = len(a)
        index_b = len(b)

        for row in range(index_b, -1, -1):
            for cell in range(index_a, -1, -1):
                directions = {}
                directions['top'] = \
                    table_alignments[row - 1][cell], [row - 1, cell]
                directions['side'] = \
                    table_alignments[row][cell - 1], [row, cell - 1]
                directions['diagonal'] = \
                    table_alignments[row - 1][cell - 1], [row - 1, cell - 1]
                directions =\
                    {k: v for k, v in directions.items() if
                     v[0] <= table_alignments[row][cell] and
                     all(val >= 0 for val in v[1])}
                for value, direction in directions.values():
                    delta = table_alignments[row][cell] - table_alignments[direction[0]][direction[1]]
                    table[direction[0]][direction[1]] = \
                        max(
                            table[direction[0]][direction[1]],
                            table[row][cell] + delta)
        return table

    @staticmethod
    def longest_sequence(X, Y):
        table = [[list() for _ in range(len(X))]
                 for _ in range(len(Y))]
        for row, _ in enumerate(table):
            table[row][0] = [(0, row)]
        for cell, _ in enumerate(table[0]):
            table[0][cell] = [(cell, 0)]
        for row, _ in enumerate(table):
            for cell, _ in enumerate(table[row]):
                if X[cell] == Y[row] and row - 1 >= 0 and cell - 1 >= 0:
                    table[row][cell] = table[row - 1][cell - 1]
                    table[row][cell] = [(c, r) for (c, r) in table[row][cell] if c != cell]
                    table[row][cell].append((cell, row))
                elif len(table[row][cell - 1]) > len(table[row - 1][cell]) and cell - 1 >= 0 and row - 1 >= 0:
                    table[row][cell] = table[row][cell - 1]
                elif cell - 1 >= 0 and row - 1 >= 0:
                    table[row][cell] = table[row - 1][cell]
        # result = [Y[cell] for cell, row in table[-1][-1]]
        for row, _ in enumerate(table):
            for cell, _ in enumerate(table[row]):
                table[row][cell] = [X[row] for row, cell in table[row][cell]]
                for item, _ in enumerate(table[row][cell]):
                    table[row][cell] = ''.join(table[row][cell])
        return table

    @classmethod
    def count_inversions(cls, l: List[int]) -> int:
        _, count = cls.split_lists(l)
        return count

    @classmethod
    def split_lists(cls, a: List[int], count=0) -> Tuple[List[int], int]:
        if len(a) < 2:
            return a, count
        half = len(a) // 2
        merge, c = cls.merge_list(a[:half], a[half:])
        count += c
        return merge, count

    @classmethod
    def merge_list(cls, a, b) -> Tuple[List[int], int]:
        print(f'a: {a}; b: {b}')
        a, count_a = cls.split_lists(a)
        b, count_b = cls.split_lists(b)
        print(f'{count_a}, {count_b}')
        count = count_a + count_b
        last_a = len(a)
        last_b = len(b)
        index_a = 0
        index_b = 0
        temp = []
        while index_a < last_a and index_b < last_b:
            if a[index_a] <= b[index_b]:
                temp.append(a[index_a])
                index_a += 1
                continue
            count += (last_a - index_a)
            temp.append(b[index_b])
            index_b += 1
        for i in range(index_a, last_a):
            temp.append(a[i])
        for i in range(index_b, last_b):
            temp.append(b[i])
            print(f'{temp}, {count}')
        return temp, count


class GraphColoring:

    def __init__(self, matrix: List[List[int]]):
        self.matrix = matrix
        self.length = len(matrix)
        for row in self.matrix:
            if len(row) != self.length:
                raise ValueError("The given matrix must be square.")
        # The color each vertex will be assigned to.
        self.colors = [0] * self.length
        # The total number of colors we will use.
        self.num_colors = 2

    def can_insert(self, row: List[int], color: int) -> bool:
        for index in range(self.length):
            if row[index] == 1 \
                    and self.colors[index] == color:
                return False
        return True

    def _colorize_graph(self, current_row: int) -> bool:
        if current_row == self.length:
            return True

        for color in range(1, self.num_colors + 1):
            if self.can_insert(self.matrix[current_row], color):
                self.colors[current_row] = color
                if self._colorize_graph(current_row + 1):
                    return True
                self.colors[current_row] = 0
        if current_row == 0:
            self.num_colors += 1
            self.colors = [0] * self.length
            self._colorize_graph(current_row)
        return False

    def colorize_graph(self) -> Union[bool, List[int]]:
        self._colorize_graph(0)
        return self.colors


class melange:

    best = []
    the_a = ''
    the_b = ''

    def __new__(cls, a: str, b: str) -> List[str]:
        cls.best = []
        cls.the_a = a
        cls.the_b = b
        a, b = list(a), list(b)
        cls._melange(a)
        complement = cls._filter(cls.best, a)
        return cls.best, complement

    @classmethod
    def _test(cls, a: List[str], short=False) -> bool:
        if not cls._contains(cls.the_b, a):
            return False
        if short:
            return True
        l = cls._filter(a, cls.the_b)
        return cls._contains(cls.the_a, l)

    @classmethod
    def _filter(cls, a, b):
        a = deque(a)
        l = []
        for char in b:
            if len(a) == 0 or char != a[0]:
                l += char
            else:
                try:
                    a.popleft()
                except IndexError:
                    pass
        return l

    @staticmethod
    def _contains(a: Iterable[str], b: Iterable[str]) -> bool:
        b = deque(b)
        for char in a:
            if not b:
                return True
            if b[0] == char:
                b.popleft()
        return not b

    @classmethod
    def _melange(cls, a: List[str], pre=None) -> Generator[Optional[List[str]], None, None]:
        if cls.best:
            return
        print(f'{pre} :: {a}')
        if pre is None:
            pre = []
        elif cls._test(pre):
            print('\nNEW BEST\nNEW BEST\n')
            cls.best = pre
        if len(a) == 1:
            candidate = pre + a
            if cls._test(candidate):
                cls.best = candidate
                return
        pre_test = cls._test(pre + a[0:1], True)
        for start in range(1, len(a)):
            if pre_test:
                cls._melange(a[start:], pre + a[0:1])
            cls._melange(a[start:], pre)












