import numpy as np

A = np.array([10,20,30,40,501])
B = np.array([5,4,3,2,1])

print(A + B)
print(A - B)
print(A * B)
print(A / B)
print(A.min() )
print(A.max() )
print(A.sum() )
c = np.dot(A, B)
print(c) 
A_reshaped = A.reshape(1,5)
print(A_reshaped)