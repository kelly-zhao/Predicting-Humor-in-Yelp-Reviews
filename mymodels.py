from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

## Logistic Regression

def log_reg_model(X_train, X_test, y_train):
    ''' fits a logistic regression model on the data and computes 
    predicted values '''

    lr = LogisticRegression(random_state=0, max_iter=500)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:,1]
    return y_pred_lr, y_proba_lr


## Naive Bayes

def naive_bayes_model(X_train, X_test, y_train):
    ''' fits a naive bayes model on the data and computes 
    predicted values '''

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    y_proba_nb = nb.predict_proba(X_test)[:,1]
    return y_pred_nb, y_proba_nb


## Random Forest

def random_forest_model(X_train, X_test, y_train):
    ''' fits a random forest model on the data and computes 
    predicted values '''

    rf = RandomForestClassifier(n_estimators = 500, # the more trees the better
                                max_depth = 3, # play around with diff depths
                                max_features = 'auto', 
                                random_state=0)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]
    return y_pred_rf, y_proba_rf



## Gradient Boosted Logistic Regression

def gradient_boosting_model(X_train, X_test, y_train):
    ''' fits a gradient boosting model on the data and computes 
    predicted values '''

    gbc = GradientBoostingClassifier(learning_rate=0.1, # trade-off with n_estimators
                                    n_estimators=500, # boosting stages to perform, more usually better
                                    max_depth=3, # tune for performance 
                                    max_features='auto',
                                    random_state=0
                                    ) 
    gbc.fit(X_train, y_train)
    y_pred_gbc = gbc.predict(X_test)
    y_proba_gbc = gbc.predict_proba(X_test)[:,1]
    return y_pred_gbc, y_proba_gbc
