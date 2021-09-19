import numpy as np

class Perceptron:
  def __init__(self,eta,epochs):
    self.weights = np.random.randn(3) * 1e-4
    print(f"Initial weights before training:{self.weights}")
    self.eta  = eta
    self.epochs = epochs
  def activationFunction(self,inputs,weights):
    Z = np.dot(inputs,weights)
    Y_hat = np.where( Z <=0,0,1)
    return Y_hat
  def fit(self,X,Y):
    self.X = X
    self.Y = Y
    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    print(f"x with bias:{X_with_bias}")
    for e in range(self.epochs):
      print(f'******* for epoch#{e} *************')
      print(f'Weigths in FPass == {self.weights}')
      self.Y_hat = self.activationFunction(X_with_bias,self.weights)
      self.error = Y - self.Y_hat
      self.total_loss()
      self.weights = self.weights + ((self.eta) * np.dot(X_with_bias.T, self.error))
      print(f'Corrected weights == {self.weights}')
  def predict(self,x):
    X_with_bias = np.c_[x, -np.ones((len(x), 1))]
    prediction = self.activationFunction (X_with_bias,self.weights)
    return prediction
  def total_loss(self):
   total_loss = np.sum(self.error)
   print(f'erro is {self.error}')
   print(f'loss is {total_loss}')