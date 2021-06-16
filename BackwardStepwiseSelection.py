import numpy as np
import csv

X1 = np.array([])
X2 = np.array([])
X3 = np.array([])
X4 = np.array([])
X5 = np.array([])
X6 = np.array([])
Y = np.array([])

predi = np.array([])
scores = np.array([])
deleted = np.array([])
predicted_y = np.array([])


# I took my methods from my labs which i wrote


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
        X1 = np.append(X1, float(row[1]))
        X2 = np.append(X2, float(row[2]))
        X3 = np.append(X3, float(row[3]))
        X4 = np.append(X4, float(row[4]))
        X5 = np.append(X5, float(row[5]))
        X6 = np.append(X6, float(row[6]))
        if row[7] == '':
            continue
        Y = np.append(Y, int(row[7]))


matrix_first = np.column_stack((np.ones(len(X1)), X1, X2, X3, X4, X5, X6))
matrix = np.delete(matrix_first,range(100,120),0)
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


matrix_second = np.column_stack((np.ones(len(X1)), X3, X5, X6))
matrix_2 = np.delete(matrix_second, range(100, 120), 0)
predictions = matrix_second[100:]
ycap2 = calculate_ycap(matrix_2, find_coef(matrix_2, Y))
rsq_2 = rsqr(Y, ycap2)
arsq_2 = adjs_rsqr(Y, ycap2,len(matrix_2[0])-1)
for row in predictions:
    predi = np.append(predi, row[1])
coef = calculate_ycap(matrix_2, find_coef(matrix_2, Y))


for row in predictions:
    ypred = (coef[0]*row[0])+(coef[1]*row[1])+(coef[2]*row[2])+(coef[3]*row[3])
    predicted_y = np.append(predicted_y, ypred)

print("R2 FOR MULTIPLE LINEAR", rsq_2)
print("Adjusted R2 FOR MULTIPLE LINEAR", arsq_2)
print("Predictions for last 20 Y values")
print(predicted_y)