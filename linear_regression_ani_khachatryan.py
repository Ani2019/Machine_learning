from typing import List, Tuple
import numpy as np

def simple_linear_regression(X, Y)-> Tuple[float, float]:
    m_X = np.mean(X)
    m_Y = np.mean(Y)
    w_1 = np.sum((X - m_X)*(Y - m_Y))/np.sum((X - m_X)**2)
    w_0 = m_Y - w_1*m_X
    return [w_0, w_1]

def linear_regression(X, Y)-> List:
    A = X.T.dot(X) 
    B = np.linalg.inv(A)
    C = B.dot(X.T) 
    W = C.dot(Y)
    return  W
 





    
if __name__ == "__main__":
   #w_best = linear_regression (np.array([[np.random.uniform(0, 1), np.random.uniform(0, 1)]
                  #for i in range(500)]), np.arange(500))
   
   w_best = linear_regression(np.array([[1,2,3],[3,4,5],[5,7,8]]),np.array([1,2,3]))
   print(w_best)
