import numpy as np
import time 

"""
Numpy is a library that extends the base capabilities of python to add a richer data
set including more numeric types, vectors, matrices and many matrix functions. 
Python and numpy can work together fairly seamlessly.
Vectors: are ordered arrays of numbers. In notation are represented with lowercase
bold letters ex: *x*. the elements of a vector are all the same type. The number of
elements in the array if often referred to as the dimension of the vector.
in math setting the index are usually from 1 to n, but in programming
they usually start from 0 to n-1.
Arrays: Numpy`s basic data structure is and indexable, n-dimensional array 
containing elements of the same type(dtype) Here dimension refers to the number
of indexes of an array. A 1-dimensional 1-D array has one index, in course 1, we will
represent vectors as NumPy 1-D arrays.
- 1-D array, shape(n,): n elements indexed from 0 to n-1
vector creation: Data creation in numpy will generally have a first parameter
which is the shape of the object. can either be a single value or a tuple of values
"""
# NumPy routines which allocate memory and fill arrays with values
a = np.zeros(4); print(f"np.zeros(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")  
a = np.zeros((4,)); print(f"np.zeros((4,)): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routine which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.); print(f"np.arange(4.): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4); print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")   

# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# These have all created a one-dimensional vector a with four elements. a.shape returns the dimensions. Here we see a.shape = (4,) indicating a 1-d array with 4 elements.

# vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

# acess an element
print(f"a[2].shape: {a[2].shape} a[2] ={a[2]}, Acessing an element returns a scalar")

# acess the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

# index must be within the range of the vector or the will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message is:")
    print(e)

# vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

# acess 5 consecutive elements (start:stop:step)
c = a[2:7:1]; print("a[2:7:1] = ", c)

# acess 3 elements separeted by two
c = a[2:7:2]; print("a[2:7:2] = ", c)

# acess all elements index 3 and above
c = a[3:]; print("a[3]    =", c)

# acess all elements below index 3
c = a[:3]; print("a[:3]   =", c)

# acess all elements
c = a[:]; print("a[:]    =", c)

# single vector operations
a = np.array([1,2,3,4])
print(f"a         = {a}")   

# negate elements of a 
b = -a
print(f"b = -a   :{b}")

# sum all elements of a, returns a scalar
b = np.sum(a)
print(f"b = np.sum(a): {b}")

# compute the mean of elements of a, returns a scalar
b = np.mean(a)
print(f"b = np.mean(a): {b}")   

# compute the elements squared, returns a scalar
b = a**2
print(f"b = a**2: {b}") 

# vector vector element-wise operations
a = np.array([1,2,3,4])
b = np.array([-1,-2,3,4])
print(f'Binary operators work element wise: {a+b}')

# for it to work the vectors must be the same size, so lets try with different sizes
c = np.array([1,2])
try:
    d = a + c
except Exception as e:
    print("The error message is:")
    print(e)

# scalar vector operations
a = np.array([1,2,3,4])
# multiply a by a scalar
b = 5 * a
print(f"b = 5 * a: {b}")    

# vector dot product: The dot product multiplies the values in two vectors element-wise and then sums the result. Vector dot product requires the dimensions of the two vectors to be the same.
def my_dot(a,b):
    """
    compute the dot product of two vectors
    Args:
       a (ndarray (n,)): input vector
       b (ndarray (n,)): input vector with same dimension as a
       returns: 
       x (scalar) 
    """
    x =0
    for i in range(a.shape[0]):
        x = x + a[i]*b[i]
    return x

# test 1-D
a = np.array([1,2,3,4])
b = np.array([-1,4,3,2])
print(f"my_dot(a,b) = {my_dot(a,b)}")
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

# We utilized the NumPy library because it improves speed memory efficiency. Let's demonstrate:
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory

"""
NumPy makes better use of available data parallelism in the underlying hardware. 
GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel. 
This is critical in Machine Learning where the data sets are often very large.
Its important in this course cause:
Going forward, our examples will be stored in an array, X_train of dimension (m,n). This will be explained more in context, 
but here it is important to note it is a 2 Dimensional array or matrix (see next section on matrices).
w will be a 1-dimensional vector of shape (n,).
we will perform operations by looping through the examples, extracting each example to work on individually by indexing X.
For example:X[i]
X[i] returns a value of shape (n,), a 1-dimensional vector. Consequently, operations involving X[i] are often vector-vector
"""
"""
Matrices 
Abstract: are two dimensional arrays. The elements of a matrix are all of the same type
In notation matrices are denoted with capitol, bold letters such as "X". in this and other labs, m
is often the number of rows and n the number of columns. The elements of a matrix can be referenced
with a two dimensional index. 
Arrays: Numpy`s basic data structure is and indexable, n-dimensional array
containing elements of the same type(dtype). These were described earlier.
Matrices hava a two-dimensional (2-D) index[m,n]
in C1 matrices are used to hold training data. m examples by n features, creating an (m,n)array
C1 does not do operations on matrices directly, but typically extracts an example as a vector and operates on that
"""

# Matrix creation
a = np.zeros((1,5))
print(f"a shape = {a.shape}, a = {a}")
a = np.zeros((2,1))
print(f"a shape = {a.shape}, a = {a}")
a = np.random.random_sample((1,1))

# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5],[4],[3]]); print(f"a shape = {a.shape}, np.array a = {a}")
a = np.array([[5]  # one can also
                ,[4] # separate values
                ,[3]]); # into separate rows
print(f"a shape = {a.shape}, np.array a = {a}")

# vector indexing operations on matrices 
a = np.arange(6).reshape((-1,2)) #reshape is a convenient way to create matrices
"""
This line of code first created a 1-D Vector of six elements. 
It then reshaped that vector into a 2-D array using the reshape command. This could have been written:
a = np.arange(6).reshape(3, 2) 
To arrive at the same 3 row, 2 column array. The -1 argument tells the routine to compute the number of rows given the size of 
the array and the number of columns.
"""
print(f"a.shape: {a.shape}, \na= {a}")

# acess an element
print(f'\na[2,0].shape: {a[2,0].shape}, a[2,0] = {a[2,0]}, type(a[2,0]) = {type(a[2,0])} Acessing an element returns a scalar\n')

# acess a row
print(f'a[2].shape: {a[2].shape}, a[2] = {a[2]}, type(a[2]) = {type(a[2])}')

#vector 2-D slicing operations
a = np.arange(20).reshape(-1,10) # 2-D array with 2 rows and 10 columns
print(f'a = \n{a}')

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")