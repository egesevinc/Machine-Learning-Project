import numpy as np
import csv


X3 = np.array([])
X5 = np.array([])
X6 = np.array([])
Y = np.array([])

adjs = np.array([])
predi1 = np.array([])
predi2 = np.array([])
predi3 = np.array([])

# These are my methods from my labs which i wrote


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


matrix_final = np.column_stack((np.ones(len(X3)), X3, X5, X6, X3*X5, X3*X6, X5*X6, X3*X3, X5*X5, X6*X6, (X3*X5)*X6))
matrix_final_fin = np.delete(matrix_final, range(100, 120), 0)
predictions = matrix_final_fin[100:]
for row in predictions:
    predi1 = np.append(predi1, row[1])

ycapfin = calculate_ycap(matrix_final_fin, find_coef(matrix_final_fin, Y))
rsq_fin = rsqr(Y, ycapfin)
arsq_fin = adjs_rsqr(Y, ycapfin, len(matrix_final_fin[0])-1)
print("R2 AFTER DIMENSION INCREASE", rsq_fin)
print("ADJUSTED R2 AFTER DIMENSION INCREASE", arsq_fin)

k = 5
print('Folds: ', k)
foldsize = int(len(matrix_final_fin) / k)

for i in range(0, len(matrix_final_fin), foldsize):
    x_test = matrix_final_fin[i:i + foldsize]
    y_test = Y[i:i + foldsize]
    x_train = np.delete(matrix_final_fin, range(i, i + foldsize), 0)
    y_train = np.delete(Y, range(i, i + foldsize), 0)

    coef = find_coef(x_train, y_train)
    ypred1 = np.dot(x_test, coef)
    for i in range(len(ypred1)):
        if ypred1[i] < 0:
            ypred1[i] = 0

    j = adjs_rsqr(y_test, ypred1, len(x_test[0]) - 1)
    adjs = np.append(adjs, j)
maxi = np.max(adjs)
for i in range(len(adjs)):
    if adjs[i] == maxi:
        print("This fold gives the max adjusted R2 value... index[]: ", i)
        fold_num = i

fold_num = (fold_num * foldsize)
print("Best fold for test data begins at: ", fold_num)
x_test = matrix_final_fin[fold_num:fold_num + foldsize]
y_test = Y[fold_num:fold_num + foldsize]
print("Ends at: ", fold_num + foldsize)
x_train = np.delete(matrix_final_fin, range(fold_num, fold_num + foldsize), 0)
y_train = np.delete(Y, range(fold_num, fold_num + foldsize), 0)
for row in x_test:
    predi2 = np.append(predi2, row[1])
for row in x_train:
    predi3 = np.append(predi3, row[1])

coef = find_coef(x_train, y_train)
ypred1 = np.dot(x_test, coef)
for i in range(len(ypred1)):
    if ypred1[i] < 0:
        ypred1[i] = 0
ypred2 = np.dot(predictions, coef)
for i in range(len(ypred2)):
    if ypred2[i] < 0:
        ypred2[i] = 0
rsq = rsqr(y_test, ypred1)
arsq = adjs_rsqr(y_test, ypred1, len(matrix_final_fin[0])-1)
print('RSQ:', rsq)
print('Adjusted R Square:', arsq)

