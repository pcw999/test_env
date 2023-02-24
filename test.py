import sys

a = b'[[[447, 441], [482, 412]]]'
b = 'asdf'
c = b.encode()

print(sys.getsizeof(a))
print(sys.getsizeof(b))
print(sys.getsizeof(c))