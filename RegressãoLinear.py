#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# In[4]:


x = [-1, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 1]
y = [-1.13, -0.57, -0.2, 0.5, 0.49, 1.49, 1.64, 2.17, 2.64, 2.95]


# In[5]:


plt.figure(figsize=(10,5))
plt.plot(x,y,'o-', label="dados originais")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()


# In[6]:


x,y = np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)


# In[7]:


reg = LinearRegression()
reg.fit(x,y)


# In[11]:


print("a estimado: ",reg.coef_.ravel()[0])
print("b estimado: ",reg.intercept_[0])


# In[10]:


y_pred = reg.predict(x)


# In[11]:


score = reg.score(x, y)


# In[16]:


plt.figure(figsize=(10,5))
plt.plot(x,y,'o', label="dados originais")
plt.plot(x,y_pred, label="regressão linear (R2: {:.3f})".format(score))
plt.hlines(y=y.mean(), xmin = x.min(), xmax = x.max(), linestyle='dashed', label="Modelo de referencia do $R^2$")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()


# In[17]:


def mse(y_true, y_pred, is_ref = False):
    if is_ref: 
        mse = ((y_true - y_true.mean())**2).mean() #Calculo do mse modelo
    else:
        mse = ((y_true - y_pred)**2).mean()
    return mse


# In[18]:


def r2(mse_reg, mse_ref):
    return 1 - mse_reg / mse_ref


# In[19]:


print("y_true: ", y.ravel())
print("y_pred: ", y_pred.ravel())


# In[20]:


mse_reg = mse(y_true = y, y_pred = y_pred)
print("MSE do modelo de regressao: ", mse_reg)
mse_ref = mse(y_true = y, y_pred = y_pred, is_ref = True)
print("MSE do modelo de referencia: ", mse_ref)


# In[21]:


r2_score = r2(mse_reg = mse_reg, mse_ref = mse_ref)
print("Coeficiente R2 do modelo implementado (calculado): ", r2_score)


# In[23]:


r2_score_skl = reg.score(x,y)
print("Coeficiente R2 do modelo implementado (scikit-learn): ", r2_score_skl)


# In[ ]:




