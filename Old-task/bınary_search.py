import random


class BinaryHeap(object):
    def __init__(self, arr=None):
        self._list = [0]
        if arr:
            for value in arr:
                self.insert(value)

    def insert(self, value):
        self._list.append(value)
        self._bubble_up(len(self._list) - 1)

    def peek_min(self):
        if len(self._list) == 1:
            raise ValueError('Empty')
        return self._list[1]

    def extract_min(self):
        if len(self._list) == 1:
            raise ValueError('Empty')
        value = self._list[1]
        self._swap(1, -1)
        self._list = self._list[:-1]
        self._bubble_down(1)
        return value

    def is_empty(self):
        return len(self._list) == 1

    def __len__(self):
        return len(self._list) - 1

    def __iter__(self):
        yield from iter(self._list[1:])

    def _swap(self, idx1, idx2):
        temp = self._list[idx1]
        self._list[idx1] = self._list[idx2]
        self._list[idx2] = temp

    def _bubble_down(self, idx):
        while 2 * idx < len(self._list):  # has at least one child
            if len(self._list) == 2 * idx + 1:
                min_child = 2 * idx
            else:
                if self._list[2 * idx] < self._list[2 * idx + 1]:
                    min_child = 2 * idx
                else:
                    min_child = 2 * idx + 1
            self._swap(min_child, idx)
            idx = min_child

    def _bubble_up(self, idx):
        parent = idx // 2
        while idx > 1 and self._list[idx] < self._list[parent]:
            self._swap(parent, idx)
            idx = parent
            parent = idx // 2


def test_heap():
    bh = BinaryHeap()
    values = random.sample(range(-15, 15), 30)
    for v in values:
        bh.insert(v)
        print(list(bh))

    for v in iter(bh):
        print(v)


test_heap()