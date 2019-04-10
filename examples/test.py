class A(object):
    def __init__(self):
        self.a = 2

class B(A):
    def __init__(self):
        super().__init__()
        self.b = 3

    def c(self):
        return self.__class__()

b = B()
c = b.c()
print(c.b)
