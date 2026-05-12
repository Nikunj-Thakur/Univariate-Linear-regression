import numpy as np

def mean(arr):
    sum=0
    for ele in arr:
        sum=sum+ele
    return (sum/len(arr))


def mean_difference(arr, mean):
    diff=np.zeros(len(arr))
    for i in range(len(arr)):
        diff[i] = arr[i] - mean
    return diff
    

def muliply_mean_differences_and_sum(arrX,arrY):
    result=0
    for i in range(len(arrX)):
        result=result + (arrX[i]*arrY[i])
    return result

def mean_diff_square(arr):
    mean_sq=0
    for i in range(len(arr)):
         mean_sq=mean_sq+(arr[i] * arr[i])
    return mean_sq

def get_slope(x_train,y_train):
    x_mean = mean(x_train)
    y_mean = mean(y_train)
    x_mean_diff = mean_difference(x_train, x_mean)
    y_mean_diff = mean_difference(y_train, y_mean)
    x_y_mean_difference_product_sum = muliply_mean_differences_and_sum(
        x_mean_diff, y_mean_diff)
    x_mean_difference_square = mean_diff_square(x_mean_diff)
    return x_y_mean_difference_product_sum/x_mean_difference_square


def get_intercept(slope,x_train,y_train):
    x_mean = mean(x_train)
    y_mean = mean(y_train)
    return y_mean-(slope)*x_mean


def calculate_predicted_values(slope, intercept,x_train,y_train):
    y_hat = np.zeros(len(x_train))
    for i in range(len(x_train)):
        y_hat[i] = intercept+(slope) * x_train[i]
    return y_hat