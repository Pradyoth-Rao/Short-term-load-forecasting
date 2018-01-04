#The Program implements the ARIMA time serie model. 

from pandas import read_csv
from pandas import datetime
import math
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

#def parser(x):
	#return datetime.strptime('190'+x, '%Y-%m')



series = read_csv('loaddata.txt', usecols=[3], engine='python', skipfooter=3)
X = series.values
print(len(series),len(X))
size = int(len(X) * 0.80)
print(size)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
print(len(test))
for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('time =%f predicted=%f, expected=%f' % (t,yhat, obs))
error = mean_squared_error(test, predictions)

# calculate root mean squared error only on the test set
testScore = math.sqrt(mean_squared_error(test, predictions))
print('Test Score: %.2f RMSE' % (testScore))

#error computation
summation = 0

for i in range(len(test)):
    summation = summation + ((test[i]-predictions[i])/test[i])

n=len(test)
accuracy = 100-((1/n)*summation*100)
print("The Prediction accuracy is:%f" % accuracy)

print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()



