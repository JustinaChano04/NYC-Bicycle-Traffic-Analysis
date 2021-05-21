import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from pylab import * 


def initialize_data(filename, iData, oData):
    
    #Import Data
    bicycle_data = pd.read_csv(filename)

    #create numpy arrays from pandas dataframe
    Y = bicycle_data[[oData]]
    y = Y.to_numpy()
    x = bicycle_data[[iData]]
    X = x.to_numpy()

    #convert strings to ints
    dimensions = y.shape
    for row in range(dimensions[0]):
        for col in range(dimensions[1]):
            num = y[row][col]
            y[row][col] = float(num.replace(',',''))

    return X, Y
def initialize_data_both(filename, bike, totalbike):
    
    #Import Data
    bicycle_data = pd.read_csv(filename)

    #create numpy arrays from pandas dataframe
    y = bicycle_data[[totalbike]]
    y = y.to_numpy()
    x = bicycle_data[[bike]]
    x = x.to_numpy()

    #convert strings to ints
    dimensions = y.shape
    for row in range(dimensions[0]):
        for col in range(dimensions[1]):
            num = y[row][col]
            y[row][col] = float(num.replace(',',''))

    dimensions = x.shape
    for row in range(dimensions[0]):
        for col in range(dimensions[1]):
            num = x[row][col]
            x[row][col] = float(num.replace(',',''))
    
    return x, y
def initialize_precipitation(filename, x, y):
        
    #import
    bicycle_data = pd.read_csv(filename)
    
    #create numpy array for bikes
    y = bicycle_data[[y]]
    y = y.to_numpy()
    #create numpy array for precipitation
    x = bicycle_data[[x]]
    x = x.to_numpy()
    
    #Parse y data, read as strings as of now, convert to int
    dimensions = y.shape
    for row in range(dimensions[0]):
        for col in range(dimensions[1]):
            num= y[row][col]
            num =  num.replace('(S)', '')
            num = num.replace('T', '0')
            y[row][col] = float(num)
    dimensions = x.shape
    for row in range(dimensions[0]):
        for col in range(dimensions[1]):
            num = x[row][col]
            x[row][col] = float(num.replace(',',''))
    return x, y

def initializeRegression(filename):
    data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    tempHigh = data[['High Temp (°F)']]
    total = data[['Total']]

    tempHigh = tempHigh.to_numpy()
    total = total.to_numpy()
    dimensions = total.shape
    for row in range(dimensions[0]):
        for col in range(dimensions[1]):
            num = total[row][col]
            total[row][col] = float(num.replace(',',''))
    
    #converting numpy array to be used later
    tempHigh = tempHigh.flatten()
    total = total.flatten()
    tot = []
    tmp = []
    for bike in total:
        tot.append(bike)
    for deg in tempHigh:
        tmp.append(deg)

    slope, intercept, r_value, p_value, std_err = stats.linregress( tmp, tot)
    return slope, intercept, r_value

def plotsQ1():
    ##### initialize data to be manipulated/graphed later
    temp1, BB = initialize_data(filename, 'High Temp (°F)', 'Brooklyn Bridge')
    temp2, MB = initialize_data(filename, 'High Temp (°F)', 'Manhattan Bridge')
    temp3, WB = initialize_data(filename, 'High Temp (°F)', 'Williamsburg Bridge')
    temp4, QB = initialize_data(filename, 'High Temp (°F)', 'Queensboro Bridge')    
    temp5, tot = initialize_data(filename, 'High Temp (°F)', 'Total')
    m,b, r = initializeRegression(filename)
    
    
    ##### plot temperature vs number of bicycles
    figure(1)
    plt.plot(np.array(temp4), np.array(QB), 'o', color='navy', markersize=2, label='QB');
    plt.plot(np.array(temp3), np.array(WB), 'o', color='darkblue', markersize=2, label='WB');
    plt.plot(np.array(temp2), np.array(MB), 'o', color='cornflowerblue', markersize=2, label='MB');
    plt.plot(np.array(temp1), np.array(BB), 'o', color='lightsteelblue', markersize=2, label='BB');
    plt.legend(markerscale=6)
    plt.xlabel("Temperature (˚F)", fontsize = 8, fontweight='bold') 
    plt.ylabel("Number of Bicycles", fontsize = 8, fontweight='bold') 
    plt.title("Bike Traffic vs Temperature (April-October)",  fontsize = 16) 
    ##### plot temperature vs total number of bicycles
    figure(2)
    plt.plot(np.array(temp5), np.array(tot), 'o', color='darkblue', markersize=2, label='plot');
    plt.plot(temp1, m * temp1 + b, color='black', label='line')
    plt.xlabel("Temperature (˚F)", fontsize = 8, fontweight='bold') 
    plt.ylabel("Total Number of Bicycles", fontsize = 8, fontweight='bold') 
    plt.title("Total Bike Traffic vs Temperature (April-October)",  fontsize = 16) 
    plt.legend(("plot", "line"), markerscale = 4)
    print(f'y = {m}x {b} r_value: {r}')
    plt.show()

def ridge_regression_model(filename):
    bike, prec = initialize_precipitation(filename, 'Total', 'Precipitation')
    
    [Xtrain, Xtest, Ytrain, Ytest] = train_test_split(bike, prec, test_size=0.25, random_state=101)

    #normalize data
    std_prec = np.std(prec)
    mean_prec = np.mean(prec)
    trn_std = np.std(bike)
    trn_mean = np.mean(bike)

    Ytrain = (Ytrain - mean_prec) / std_prec    
    Xtrain = (Xtrain - trn_mean) / trn_std
    Xtest = (Xtest - trn_mean) / trn_std
    
    #define lambda x axis 
    lmbda = [0]
    lmbda = np.array(lmbda)
    lmbda = np.append(lmbda,np.logspace(-1.00,2.00, num = 101, base = 10 )) 

    _model = []
    _mse = []

    #Define the range of lambda to test
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model =  linear_model.Ridge(alpha = l , fit_intercept = True)

        #Evaluate the MSE on the test set
        model.fit(Xtrain, Ytrain)
        norm_y_pred_test = model.predict(Xtest)
        y_pred_test = (norm_y_pred_test * std_prec) + mean_prec
        mse = mean_squared_error(Ytest, y_pred_test)

        _model.append(model)
        _mse.append(mse)

    #plot MSE as a function of lambda
    figure(3)
    plot_ridge_regression(lmbda, _mse)
    
    #determine the index of the lowest mse to use for hypothesis testing
    i = _mse.index(min(_mse))
    lmbda_save = lmbda[i]
    mse_save = _mse[i]
    model_save = _model[i]

    return model_save, Ytest, y_pred_test, bike, trn_mean


def plot_ridge_regression(lmbda, MSE):
    plt.plot(lmbda, MSE, color = 'navy', linewidth = 2)
    plt.title('MSE vs. Lambda')
    plt.xlabel('Regularization Parameter Lambda')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

def hypothesis_test(filename):
    
    #find lowest MSE value
    _model, y_test, y_pred_test, X, trn_mean = ridge_regression_model(filename)
    mean = _model.coef_[0][0]

    #Determine Standard Error, SE
    SE = (float(np.sqrt(np.sum((y_test - y_pred_test)**2) / float(np.size(y_test) - 2)))) / (float(np.sqrt(np.sum((X - trn_mean)**2))))
    
    #Test statistic z
    z = mean / SE
    #p value
    p = 2 * stats.norm.cdf(-abs(SE))

    print('\n-----Statements-----')
    print('Null Hypothesis Ho: Coefficient of regression is 0 (showing no correlation between bicycle traffic and precipitation)\n')
    print('Alternative Hypothesis H1: Coefficient of regression is NOT 0 (showing there IS correlation between bicycle traffic and precipitation)\n')
    print('-----Assumptions----')
    print('Since the number of datapoints n > 30, we can use the z-test to conduct hypothesis testing\n')
    print('-----Statistics-----')
    print(f'SE: {SE}')
    print(f'z-score: {z}')
    print(f'p-value: {p}\n')
    print('-----Conclusion-----')
    print(f'Based on the p-value of {p}, at various alpha levels of 0.01, 0.05, and 0.1, the results are INSIGNIFICANT which means we CANNOT reject the Null Hypothesis in favor of the Alternative Hypothesis meaning precipitation does not depend on bike traffic')

def poly_regression(filename):
    #initialize data
    MB, tot = initialize_data_both(filename, 'Manhattan Bridge', 'Total')
    WB, tot = initialize_data_both(filename, 'Williamsburg Bridge', 'Total')
    QB, tot = initialize_data_both(filename, 'Queensboro Bridge', 'Total')
    BB, tot = initialize_data_both(filename, 'Brooklyn Bridge', 'Total')
    #convert to list
    mb = []
    wb = []
    qb = []
    bb = []
    total = []
    for i in range(0, len(tot)):
        mb.append(float(MB[i]))
        wb.append(float(WB[i]))
        qb.append(float(QB[i]))
        bb.append(float(BB[i]))
        total.append(float(tot[i]))
    #normalize
    mb_norm =  preprocessing.normalize([np.array(mb)])
    wb_norm = preprocessing.normalize([np.array(wb)])
    qb_norm = preprocessing.normalize([np.array(qb)])
    bb_norm = preprocessing.normalize([np.array(bb)])
    total_norm = preprocessing.normalize([np.array(total)])

    print(f'MB_mean: {np.mean(mb)} WB_mean: {np.mean(wb)} QB_mean: {np.mean(qb)} BB_mean: {np.mean(bb)}')
    
    mb_norm = mb_norm.tolist()
    wb_norm = wb_norm.tolist()
    qb_norm = qb_norm.tolist()
    bb_norm = bb_norm.tolist()
    total_norm = total_norm.tolist()

    #print(f' {mb_norm[0][0]} {wb_norm[0][0]} {qb_norm[0][0]} {bb_norm[0][0]}')

    #finding the coefficients of the equation
    feature_matrix = []
    for i in range(0, len(tot)):
        temp = []
        temp.append(mb_norm[0][i])
        temp.append(wb_norm[0][i])
        temp.append(qb_norm[0][i])
        temp.append(bb_norm[0][i])
        temp.append(1) 
        feature_matrix.append(temp)
    
    B = least_squares(feature_matrix, total)
    B = B.tolist()
    print(f'Manhattan: {B[0]} Williamsburg: {B[1]} Queensboro: {B[2]} Brooklyn Bridge: {B[3]}')
    figure(4)
    plt.plot(np.array(mb), np.array(total),'o', color = "navy", markersize=2, label='Manhattan Bridge')
    plt.plot(np.array(wb), np.array(total), 'o',color = "cornflowerblue", markersize=2, label='Williamsburg Bridge')
    plt.plot(np.array(qb), np.array(total), 'o',color = "blue", markersize=2, label='Queensboro Bridge')
    plt.plot(np.array(bb), np.array(total), 'o',color = "slategrey", markersize=2, label='Brooklyn Bridge')
    plt.xlabel('Number of Bicycles')
    plt.ylabel('Total Number of Bicycles')
    plt.title("Total vs Number of Bicycles")
    plt.legend(("Manhattan Bridge", "Williamsburg Bridge", "Queensboro Bridge", "Brooklyn Bridge"),  markerscale=6)
     
    plt.show()



#find the Coefficients of the Polynomial Regression    
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)
    Xt = X.T
    trans = np.dot(Xt, X)
    inv = np.linalg.inv(trans)
    mult = np.dot(inv, Xt)
    B = np.dot(mult, y)
    return B



if __name__ == '__main__':
    filename = 'NYC_Bicycle_Counts_2016_Corrected.csv'
    ##########Q1###########
    poly_regression(filename)

    ##########Q2###########
    initializeRegression(filename)
    plotsQ1()

    ##########Q3###########
    hypothesis_test(filename)

    



