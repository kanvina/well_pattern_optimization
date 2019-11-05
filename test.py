import numpy as np
import copy
a=[10,[0,1]]
b=copy.deepcopy(a)
b[1][0]=100


print(a)
print(b)



# def dec2bin(num):
#     mid = []
#     while True:
#         if num == 0: break
#         num,rem = divmod(num, 2)
#         mid.insert(0,rem)
#
#     return mid
# print (dec2bin(7))