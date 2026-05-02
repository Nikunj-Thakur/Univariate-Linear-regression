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

def mean_square(arr):
    mean_sq=0
    for i in range(len(arr)):
         mean_sq=mean_sq+(arr[i] * arr[i])
    return mean_sq