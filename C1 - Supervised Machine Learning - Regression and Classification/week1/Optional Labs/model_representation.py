import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
})

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
# m is the number of training examples
# .shape function of numpy that retirns a python tuple with an entry for each dimension
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
# m can also be used by the function len of python
# m = len(x_train)
# we will use (x^(i), y^(i)) to denote the the i^th training example
# since python is zero indexed we start(x^(0),y^(0))
i = 1
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
# now we are gonna plot the data using scatter() function of matplotlib
# marker and c show the points as red crosses(default is blue dots)
# plot the data points
#plt.scatter(x_train, y_train, marker = 'x', c = 'r')
# set the title
#plt.title("Housing prices")
# set the y-axis label
#plt.ylabel('Price (in 1000s of dollars)')
# set the x-axis label
#plt.xlabel('Size (1000 sqft)')
#plt.show()
# the model function for linear regression: fw,b(x^(i)) = wx^(i) + b
# The formula above is how you can represent straight lines - different values of w and b give you different straight lines on the plot.
w = 200
b = 100

def compute_model_output(x,w,b):
    """
    computes the prediction of a linear model 
    Args:
        x(ndarray(m,)): Data, m examples
        w,b (scalar): model parameters
        returns 
        y ( ndarray (m,)): target values
        """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb
# now lets call the function and plot the output
tmp_f_wb = compute_model_output(x_train, w,b)
# plot our model prediction
plt.plot(x_train, tmp_f_wb, c = 'b', label = 'Our Prediction')
# plot the data points
plt.scatter(x_train, y_train, marker= 'x', c = 'r', label = 'actual values')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# now lets make a predction
x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} thousand dollars")