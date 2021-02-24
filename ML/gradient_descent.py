
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data.csv')    
df_countries = df[df['RegionCode'].isna()]
df_countries = df_countries.drop(columns=['RegionCode', 'RegionName'])

country_code = 'IT'

def outbreak(data: pd.DataFrame, threshold: int = 100):
    return data['Confirmed'] > threshold

cols = ['Date', 'CountryCode', 'CountryName', 'Confirmed']
country = df_countries[df_countries['CountryCode'] == country_code][cols]
country = country[outbreak(country)]
country['Date'] = pd.to_datetime(country['Date'])
basedate = country['Date'].iloc[0]
country['Days since outbreak'] = (country['Date'] - basedate).dt.days
country = country.set_index('Date')


X = country.iloc[:,-1].values
X = X.astype(float)
y = country.iloc[:,-2].values


def h(X, theta):
    return theta[0] / (1.0 + np.exp(-theta[1]*(X - theta[2])))

def loss(theta, X, y):
    prediction = h(X, theta)
    m = len(X)
    loss = prediction - y
    return np.sum(loss ** 2) / (2*m)


alpha = 1e-10 / 2
theta = [max(y), 1.0, np.median(X)]
losses = []

for _ in range(5000):
    denom = np.square(1.0 + np.exp(-theta[1]*(X-theta[2])))
    
    grad0 = (h(X,theta) - y) @ (h(X,theta)/theta[0])
    grad1 = (h(X,theta) - y) @ ((theta[0]*(X-theta[2])*np.exp(-theta[1]*(X-theta[2])))/denom)
    grad2 = (h(X,theta) - y) @ (-(theta[0]*theta[1]*np.exp(-theta[1]*(X-theta[2])))/denom)
     
    theta[0] = theta[0] - ((alpha/len(X)) * grad0)
    theta[1] = theta[1] - ((alpha/len(X)) * grad1)
    theta[2] = theta[2] - ((alpha/len(X)) * grad2)
    
    losses.append(loss(theta, X, y))
    
    
print('Estimated function: {0:.3f} / (1 + e^({1:.3f} * (X - {2:.3f}))'.format(*theta))

x = range(0, 5000)
plt.xlabel('Number of iterations')
plt.ylabel('Loss')
plt.title('Loss Function')
plt.plot(x, losses)
plt.show()

estimate = h(X,theta)

from scipy import optimize
def logistic(x: float, a: float, b: float, c: float):
    return a / (1.0 + np.exp(-b*(x-c)))
param, _ = optimize.curve_fit(logistic, X, y, maxfev=int(1E5), p0=[max(y), 1.0, np.median(X)])
estimate_curvefit = h(X,param)


country.plot(kind = 'bar', x = 'Days since outbreak', y = 'Confirmed', figsize=(16,8), label='Confirmed Cases')
plt.plot(country['Days since outbreak'], estimate, color='red', label='My fit')
plt.plot(country['Days since outbreak'], estimate_curvefit, color='green', label='curve_fit estimate')
plt.title(f"Covid-19 cases in {country_code}")
plt.legend()
plt.show()


