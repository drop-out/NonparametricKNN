
##What is NonparametricKNN
It is a KNN regressor that gives predictions based on customized loss function. KNN regressor in `sklearn` simply gives mean value of nearest neighbors as prediction, while NonparametricKNN will search the neighbors to find out if it could give a prediction that is better than the mean value. NonparametricKNN could significantly outperform ordinary KNN, especially when the loss function is strange (for example, SMAPE). Though grid search is time consuming, NonparametricKNN is still fairly fast.

NonparametricKNN supports:

- **Built-in loss**, Mean Squared Error loss (MSE loss or L2 loss), Mean Aboslute Error loss (MAE or L1 loss) and Symmetric Mean Absolute Percentage Error loss (SMAPE loss).

- **Customized loss**, user could use self-defined loss function as well.


##Dependence
NonparametricKNN is implemented in `Python 3.6`, using `numpy` to do vector operations. NonparametricKNN could be seen as a modified version of `sklearn.neighbors.KNeighborsRegressor`. Those packages can be easily installed using `pip`.

- [NumPy](https://github.com/numpy/numpy)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)



## Quick Start

**Import the module**
```python
from npknn import NonparametricKNN
```

**Initialize model**
```python
model = NonparametricKNN(n_neighbors=3,loss='L2')
```
For the loss function, following choices are provided:
* `'L2'`: Mean Squared Error loss (MSE)
* `'L1'`: Mean Aboslute Error loss (MAE)
* `'SMAPE'`: Symmetric Mean Absolute Percentage Error loss (SMAPE)

**Train**
```python
model.fit(train,target)
```
All inputs should be numpy arrays. `train` should be 2D array and `target` should be 1D array.

**Predict**
```python
model.predict(test)
```
Return predictions as numpy array.

**Customized loss**
```python
def loss(pred,true):
    result = np.power(pred-true,4)
    return result.mean()
model = NonparametricKNN(n_neighbors=3,loss=loss)
```
Loss function should take two numpy arrays as inputs, and return a scalar. Directly pass the loss function as argument.