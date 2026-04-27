import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import joblib

def train_models(X, y_clf, y_reg):
    """
    Train classification and regression models.
    """
    models_clf = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForestClassifier': RandomForestClassifier(random_state=42)
    }
    
    models_reg = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(random_state=42)
    }
    
    results_clf = {}
    results_reg = {}
    
    for name, model in models_clf.items():
        scores_acc = cross_val_score(model, X, y_clf, cv=5, scoring='accuracy')
        scores_f1 = cross_val_score(model, X, y_clf, cv=5, scoring='f1')
        results_clf[name] = {
            'accuracy_mean': scores_acc.mean(),
            'accuracy_std': scores_acc.std(),
            'f1_mean': scores_f1.mean(),
            'f1_std': scores_f1.std()
        }
        # Fit and save model
        model.fit(X, y_clf)
        joblib.dump(model, f'models/{name}.pkl')
    
    for name, model in models_reg.items():
        scores_r2 = cross_val_score(model, X, y_reg, cv=5, scoring='r2')
        scores_mse = cross_val_score(model, X, y_reg, cv=5, scoring='neg_mean_squared_error')
        results_reg[name] = {
            'r2_mean': scores_r2.mean(),
            'r2_std': scores_r2.std(),
            'mse_mean': -scores_mse.mean(),  # negate since it's neg_mse
            'mse_std': scores_mse.std()
        }
        # Fit and save model
        model.fit(X, y_reg)
        joblib.dump(model, f'models/{name}.pkl')
    
    return results_clf, results_reg

if __name__ == '__main__':
    df = pd.read_csv('data/bitcoin_featured.csv', index_col=0, parse_dates=True)
    
    # Features: exclude targets and raw close/volume if needed, but use engineered
    feature_cols = [col for col in df.columns if col not in ['next_return', 'direction']]
    X = df[feature_cols]
    y_clf = df['direction']
    y_reg = df['next_return']
    
    results_clf, results_reg = train_models(X, y_clf, y_reg)
    
    # Save results
    pd.DataFrame(results_clf).T.to_csv('results/classification_results.csv')
    pd.DataFrame(results_reg).T.to_csv('results/regression_results.csv')
    
    print("Models trained and results saved.")