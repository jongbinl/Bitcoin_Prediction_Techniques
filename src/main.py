import subprocess
import pandas as pd

def main():
    # Run data pipeline
    subprocess.run(['c:/Users/Jongbin/Bitcoin_Prediction_Techniques/Bitcoin_Prediction_Techniques/.venv/Scripts/python.exe', 'src/data_pipeline.py'], cwd='.')
    
    # Run feature engineering
    subprocess.run(['c:/Users/Jongbin/Bitcoin_Prediction_Techniques/Bitcoin_Prediction_Techniques/.venv/Scripts/python.exe', 'src/features.py'], cwd='.')
    
    # Run model training
    subprocess.run(['c:/Users/Jongbin/Bitcoin_Prediction_Techniques/Bitcoin_Prediction_Techniques/.venv/Scripts/python.exe', 'src/models.py'], cwd='.')
    
    # Print results
    print("Classification Results:")
    clf_results = pd.read_csv('results/classification_results.csv', index_col=0)
    print(clf_results)
    
    print("\nRegression Results:")
    reg_results = pd.read_csv('results/regression_results.csv', index_col=0)
    print(reg_results)

if __name__ == '__main__':
    main()