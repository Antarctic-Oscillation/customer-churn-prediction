# Customer Churn Model Performance Report

Generated: 2025-12-10 03:21:05.453642

## xgboost

- ROC-AUC: 0.8463
- Best Params: `{'subsample': 0.8, 'reg_lambda': 0.1, 'reg_alpha': 0.01, 'n_estimators': 200, 'min_child_weight': 1, 'max_depth': 4, 'learning_rate': 0.01, 'gamma': 0, 'colsample_bytree': 0.7}`
- CV Mean Score: 0.8474976429364943

### Classification Report

```
              precision    recall  f1-score   support

           0       0.82      0.93      0.87      1035
           1       0.70      0.44      0.54       374

    accuracy                           0.80      1409
   macro avg       0.76      0.69      0.71      1409
weighted avg       0.79      0.80      0.79      1409

```

### Confusion Matrix

```
[[964  71]
 [209 165]]
```

## logistic

- ROC-AUC: 0.8417
- Best Params: `{'C': 100.0, 'penalty': 'elasticnet', 'solver': 'saga', 'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42, 'l1_ratio': 0.5}`
- CV Mean Score: None

### Classification Report

```
              precision    recall  f1-score   support

           0       0.91      0.73      0.81      1035
           1       0.51      0.79      0.62       374

    accuracy                           0.74      1409
   macro avg       0.71      0.76      0.71      1409
weighted avg       0.80      0.74      0.76      1409

```

### Confusion Matrix

```
[[753 282]
 [ 78 296]]
```

## random_forest

- ROC-AUC: 0.8443
- Best Params: `{'oob_score': True, 'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 7, 'class_weight': 'balanced', 'bootstrap': True}`
- CV Mean Score: None

### Classification Report

```
              precision    recall  f1-score   support

           0       0.91      0.75      0.82      1035
           1       0.53      0.78      0.64       374

    accuracy                           0.76      1409
   macro avg       0.72      0.77      0.73      1409
weighted avg       0.81      0.76      0.77      1409

```

### Confusion Matrix

```
[[780 255]
 [ 81 293]]
```

## svm

- ROC-AUC: 0.8367
- Best Params: `{'kernel': 'linear', 'class_weight': 'balanced', 'C': 1.0}`
- CV Mean Score: None

### Classification Report

```
              precision    recall  f1-score   support

           0       0.91      0.66      0.76      1035
           1       0.46      0.82      0.59       374

    accuracy                           0.70      1409
   macro avg       0.69      0.74      0.68      1409
weighted avg       0.79      0.70      0.72      1409

```

### Confusion Matrix

```
[[680 355]
 [ 67 307]]
```

## neural

- ROC-AUC: 0.8452
- Best Params: `{'validation_fraction': 0.1, 'solver': 'adam', 'max_iter': 500, 'learning_rate_init': 0.01, 'learning_rate': 'adaptive', 'hidden_layer_sizes': (64, 32), 'epsilon': 1e-08, 'early_stopping': False, 'beta_2': 0.999, 'beta_1': 0.9, 'alpha': 0.001, 'activation': 'tanh'}`
- CV Mean Score: None

### Classification Report

```
              precision    recall  f1-score   support

           0       0.84      0.91      0.87      1035
           1       0.66      0.51      0.57       374

    accuracy                           0.80      1409
   macro avg       0.75      0.71      0.72      1409
weighted avg       0.79      0.80      0.79      1409

```

### Confusion Matrix

```
[[937  98]
 [184 190]]
```
