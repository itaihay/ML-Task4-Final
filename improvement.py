prob = rf_clf.predict_proba(X_train_pseudo)
pred = rf_clf.predict(X_train_pseudo)
prob_indices = prob.max(axis=1) > 0.7

y_train_pseudo[prob_indices] = pred

X_train = pd.concat([X_train_real, X_train_pseudo[prob_indices]])
y_train = np.append(y_train_real, X_train_pseudo[prob_indices])