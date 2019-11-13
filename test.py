from sympy import *


x = Symbol('x')

b=x * 2 - 4
a=solve(b, x)

print(a[0])