from numpy import *
import numpy
rr = array([[0,0,1]]).T
xn = array([[0,0,1],[1,1,1],[1,0,1]])
wn = array([[1.0,1.0,1.0]]).T
def sigmoid(x):
    s =  1/(1+exp(-x))
    return s
def _tanh_(x):
    result = exp(x)-exp(-x) / exp(x) + exp(-x) 
def softsign(x):
    if x > 0:
        result =  1 / 1 + x
    elif x < 0:
        result =  1/ 1 - x
    else:
        result = 1
while 1:
    hd = dot(xn,wn)
    tahmin = sigmoid(hd)
    wn = wn + dot(xn.T,((rr-tahmin)* tahmin * (1-tahmin)))
    print(str(numpy.mean(numpy.abs(rr-tahmin))))