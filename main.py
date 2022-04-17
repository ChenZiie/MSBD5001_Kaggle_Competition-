import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings('ignore')

luck_number = 1024
np.random.seed(luck_number)

pd.set_option('display.max_columns', None)

################read data
train = pd.read_csv('data/train.csv',index_col='id')
test = pd.read_csv('data/test.csv',index_col='id')

###############data pred
train = train.dropna(how='any', axis = 0)

test = (test-test.mean())/test.std()

features = train.loc[:, train.columns != 'label']
features = (features-features.mean())/features.std() # try change the right part to features.
train.loc[:, train.columns != 'label']  = features
X = train.drop('label', axis=1)
y = train['label']
# val = train.sample(frac=0.2)
# train = train[~train.index.isin(val.index)]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=luck_number)

##############################
tuned_parameters = [{'n_estimators': [5,10,30,50,100], 'max_depth': [3,4,5,10,20]}]
clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,
                   scoring='accuracy')
clf.fit(X, y)
best_dic = clf.best_params_
print(best_dic)
print(clf.best_score_)
# print(cross_val_score(model, X, y, cv=5, scoring="accuracy"))
# model = RandomForestClassifier(max_depth= best_dic['max_depth'], n_estimators=best_dic['n_estimators'])

model = RandomForestClassifier()

model.fit(X,y)

test['label'] = model.predict(test)
pred = test['label']
print(pred)
pred.to_csv('submission.csv')


# model = RandomForestClassifier(max_depth=5,n_estimators=50)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_val)

# print(cross_val_score(model, X, y, cv=5, scoring="accuracy"))
# print(accuracy_score(y_pred,y_val))

