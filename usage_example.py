from continuous_kfold import ContinuousSKFold
import catboost as cb
import pandas as pd


def main(X, y, model) -> pd.series:
  
  results = pd.Series(index=X.index)
  
  folds = ContinuousSKFold(X=X,
                           y=y).create_folds()
  
  for i, fold in enumerate(folds):
    X_train, X_test = X.iloc[fold['train']], X.iloc[fold['test']]
    y_train, y_test = y[fold['train']], y[fold['test']]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test, y_test)
    
    results.iloc[fold['test']] = predictions
    
  return results


if __name__ == '__main__':
  X = 
  y = 
  main(X, y, cb.CatBoostRegressor())
