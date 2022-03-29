import numpy as np  # import numpy to work with arrays
import pandas as pd  # import pandas to manipulate the dataset
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import hinge_loss, accuracy_score, log_loss
from matplotlib import pyplot as plt  # import the module matplotlib.pyplot to do visulization


# import the csv file through pandas
df = pd.read_csv('WineQT.csv')

# get rid of unnecessary columns
df.drop(columns=['Id'], inplace=True)

# rename columns to simplify handling
df.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'sugar', 'chlorides', 'FSD', 'TSD', 'density', 'pH',
              'sulphates', 'alcohol', 'quality']


# classify function allows us to quantify quality binarily, with 1 as "good" and -1 as "not good"
def classify(dataframe):
    if dataframe['quality'] >= 7:
        return 1
    else:
        return -1


# add column 'qualityBin' to the dataset
df['qualityBin'] = df.apply(classify, axis=1)


# set features as numpy array to increase accuracy rather than only using one feature
X = np.asarray(df[['fixed_acidity', 'citric_acid', 'sugar', 'chlorides', 'density', 'pH', 'alcohol']])
# set label as binary quality
y = np.asarray(df['qualityBin'])

# split data into training and validation sets
X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.80, shuffle=True, random_state=40)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.20, random_state=40)

# a higher value of c allows us to build on a model robust to outliers
c = 100000

# define classification regression models
model = SVC(C=c, kernel="linear")
model_2 = LogisticRegression(penalty='l2', C=c)
models = [model, model_2]
modelNames = ["SVC", "Logistic Regression"]


# scaler = StandardScaler()


tr_accs = {}
val_accs = {}

tr_errs = {}
val_errs = {}
test_errs = {}


# apply both models to the data to compare accuracies
for i in range(len(models)):
    # fit data to the model
    models[i].fit(X_train, y_train)

    # use predict function to train the data
    y_pred_train = models[i].predict(X_train)
    # calculate accuracy
    acc_train = accuracy_score(y_train, y_pred_train)
    tr_accs[i] = acc_train

    # use predict function on validation set
    y_pred_val = models[i].predict(X_val)
    # calculate accuracy
    acc_val = accuracy_score(y_val, y_pred_val)
    val_accs[i] = acc_val

    print("Training accuracy for " + modelNames[i] + " :", acc_train)

    print("Validation accuracy " + modelNames[i] + " :", acc_val)


print("-----------------")
model.fit(X_train, y_train)
# Hinge loss calculation for training set
y_pred_train_1 = model.predict(X_train)
tr_error_1 = hinge_loss(y_train, y_pred_train_1)
tr_errs[0] = tr_error_1
print("SVC training loss: ", tr_error_1)
# Hinge loss calculation for validation set
y_pred_val_1 = model.predict(X_val)
val_error_1 = hinge_loss(y_val, y_pred_val_1)
val_errs[0] = val_error_1
print("SVC validation loss: ", val_error_1)
y_pred_test_1 = model.predict(X_test)
test_errs_1 = hinge_loss(y_test, y_pred_test_1)
test_errs[0] = test_errs_1
print("SVC test loss :", test_errs_1)
print("------------")



# features_standardized = scaler.fit_transform(X_train)
# logistic_regression = LogisticRegressionCV(penalty='l2', Cs=1000)
# logistic_regression.fit(features_standardized, y_train)

# Logistic loss calculation for training set
model_2.fit(X_train, y_train)
y_pred_train_2 = model_2.predict_proba(X_train)
tr_error_2 = log_loss(y_train, y_pred_train_2)
tr_errs[1] = tr_error_2
print("Logistic Regression training loss: ", tr_error_2)
# Logistic loss calculation for validation set
y_pred_val_2 = model_2.predict_proba(X_val)
val_error_2 = log_loss(y_val, y_pred_val_2)
val_errs[1] = val_error_2
print("Logistic Regression validation loss: ", val_error_2)
model_2.fit(X_test, y_test)
y_pred_test_2 = model_2.predict_proba(X_test)
test_errs_2 = log_loss(y_test, y_pred_test_2)
test_errs[1] = test_errs_2
print("Logistic Regression test loss :", test_errs_2)






fig, axes = plt.subplots(1, 3, figsize=(15, 4))
x = np.arange(2)
y1 = list(tr_errs.values())
y2 = list(val_errs.values())
width = 0.2

axes[0].bar(x-0.2, y1, width, color='cyan')
axes[0].bar(x, y2, width, color='orange')
axes[0].set_ylim(0, 1)
axes[0].set_xticks(x)
axes[0].set_xticklabels(["SVC", "LogisticRegression"])
axes[0].set_ylabel("Errors")
axes[0].legend(["training error", "validation error"])

# axes[1].set_xlabel("feature(mintmp)")
# axes[1].set_yticks([0,1])
# axes[1].set_ylabel('label (class)')
# axes[1].set_title("SVC")
# axes[1].scatter(X_train[:,0],y_train,s=80,c="skyblue",label="training datapoints")
# axes[1].scatter(X_val[:,0],y_val,s=30,c="blue",label="validation datapoints")
# # axes[1].scatter(X_fit, model.predict(X_fit.reshape(-1, 1)),color='r',s=5,label='predicted label ($\hat{y}=1$ if $h(x) > 0$)')
# axes[1].legend()

# axes[2].set_xlabel("feature(mintmp)")
# axes[2].set_yticks([0,1])
# axes[2].set_ylabel('label (class)')
# axes[2].set_title("LogisticRegression")
# axes[2].scatter(X_train[:,0],y_train,s=100,c="skyblue",label="training datapoints")
# axes[2].scatter(X_val[:,0],y_val,s=30,c="blue",label="validation datapoints")
# axes[2].scatter(X_fit, model_2.predict(X_fit.reshape(-1, 1)),color='r',s=5,label='predicted label ($\hat{y}=1$ if $h(x) > 0$)')
# axes[2].legend()
plt.show()
