import numpy as np
import matplotlib.pyplot as plt
import csv

X3 = np.array([])
X5 = np.array([])
X6 = np.array([])
Y = np.array([])

adjs = np.array([])
predi1=np.array([])
predi2=np.array([])
predi3=np.array([])
scores = np.array([])
deleted = np.array([])

# These methods are from my labs which i wrote

def find_coef(x, y):

    b_hat = np.dot(x.T, x)
    b_hat = np.linalg.pinv(b_hat)
    b_hat = np.dot(b_hat, x.T)
    b_hat = np.dot(b_hat, y)
    return b_hat

def calculate_ycap(x, coef):
    y_capped = np.dot(x,coef)
    return y_capped


def rsqr(y, ycap):
    yavg = np.mean(y)
    rss = 0
    tss = 0
    for i in range(len(y)):
        rss += np.square((y[i]-ycap[i]))
        tss += np.square((y[i]-yavg))
    rsquared = 1 - (rss/tss)
    # print("R^2 score:", rsquared)
    return rsquared


def adjs_rsqr(y,ycap,d):
    yavg_adj = np.mean(y)
    rss_adj = 0
    tss_adj = 0
    for i in range(len(y)):
        rss_adj += np.square((y[i] - ycap[i]))
        tss_adj += np.square((y[i] - yavg_adj))
    adjusted_rsqr=1-((rss_adj/(len(y)-d-1))/(tss_adj/(len(y)-1)))

    return adjusted_rsqr


with open("data.csv") as f:
    mylist = list(csv.reader(f))

for row in mylist:
    if row != mylist[0]:
        X3 = np.append(X3, float(row[3]))
        X5 = np.append(X5, float(row[5]))
        X6 = np.append(X6, float(row[6]))
        if row[7] == '':
            continue
        Y = np.append(Y, int(row[7]))

Z1 = X3*X5
Z2 = X3*X6
Z3 = X5*X6
Z4 = X3*X3
Z5 = X5*X5
Z6 = X6*6
Z7 = (X3*X5)*X6
matrix_final = np.column_stack((np.ones(len(X3)), X3, X5, X6, Z1, Z2, Z3, Z4, Z5, Z6, Z7))
matrix = np.delete(matrix_final, range(100, 120), 0)

while len(matrix[0]) > 1:
    container = np.array([])
    for x in range(1, len(matrix[0]), 1):
     Matrixx = matrix
     Matrixx = np.delete(matrix, x, 1)
     ycapx = calculate_ycap(Matrixx, find_coef(Matrixx, Y))
     container = np.append(container, rsqr(Y, ycapx))

    comp = np.sort(container)
    for y in range(0, len(container), 1):
        if comp[len(container)-1] == container[y]:
            deleted=np.append(deleted, y+1)
            matrix = np.delete(matrix, y+1, 1)
            break

    ycap_final = calculate_ycap(matrix, find_coef(matrix, Y))
    scores = np.append(scores, adjs_rsqr(Y, ycap_final, len(matrix[0])-1))


print("Adjusted R2 values after each deletion of variable", scores)
print("Deleted variables index in the matrix for each iteration", deleted)
