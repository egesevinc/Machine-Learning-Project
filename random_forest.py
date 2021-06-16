import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as rf

X3 = np.array([])
X5 = np.array([])
X6 = np.array([])
Y = np.array([])

predx2 = np.array([])
predx3 = np.array([])
rsq1 = np.array([])
rsq2 = np.array([])
rsq3 = np.array([])
arsq1 = np.array([])
arsq2 = np.array([])
arsq3 = np.array([])
final_predictions_final = np.array([])


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


matrix_final_fin_f = np.column_stack((X3, X5, X6, X3 * X5, X3 * X6, X5 * X6, X3 * X3, X5 * X5, X6 * X6, (X3 * X5) * X6))
matrrrrrix = np.delete(matrix_final_fin_f, range(100, 120), 0)
predictions = matrix_final_fin_f[100:]
x_test = matrrrrrix[60:80]
y_test = Y[60:80]

x_train = np.delete(matrrrrrix, range(60, 80), 0)
y_train = np.delete(Y, range(60, 80), 0)

for i in range(1, 101):
    regression1 = rf(max_depth=10, n_estimators=i, max_features='auto')
    regression2 = rf(max_depth=10, n_estimators=i, max_features='sqrt')
    regression3 = rf(max_depth=10, n_estimators=i, max_features='log2')
    regression1.fit(x_train, y_train)
    regression2.fit(x_train, y_train)
    regression3.fit(x_train, y_train)
    pred1 = regression1.predict(x_test)
    pred2 = regression2.predict(x_test)
    pred3 = regression3.predict(x_test)
    predd = regression1.predict(predictions)
    final_predictions_final = np.append(final_predictions_final, predd, -1)
    rsq1 = np.append(rsq1, rsqr(y_test, pred1))
    rsq2 = np.append(rsq2, rsqr(y_test, pred2))
    rsq3 = np.append(rsq3, rsqr(y_test, pred3))
    arsq1 = np.append(arsq1, adjs_rsqr(y_test, pred1, 10))
    arsq2 = np.append(arsq2, adjs_rsqr(y_test, pred2, 10))
    arsq3 = np.append(arsq3, adjs_rsqr(y_test, pred3, 10))


final_predictions_final = np.reshape(final_predictions_final, (100, 20))
maxi = np.max(arsq1)

print("R2 Scores: ")
print("Auto Feature: ", np.max(rsq1))
print("SQRT Feature: ", np.max(rsq2))
print("log2 Feature: ", np.max(rsq3))
print("***********************************************")
print("Adjusted R2 Scores: ")
print("Auto Feature: ", np.max(arsq1))
print("SQRT Feature: ", np.max(arsq2))
print("log2 Feature: ", np.max(arsq3))
for i in range(len(arsq1)):
    if arsq1[i] == maxi:
        print(final_predictions_final[i])

plt.figure()
plt.plot(range(100), rsq1, "red", label='Auto')
plt.plot(range(100), rsq2, "blue", label='SQRT')
plt.plot(range(100), rsq3, "green", label='log2')
plt.xlabel("Estimators")
plt.ylabel("R2")
plt.legend()
plt.show()
