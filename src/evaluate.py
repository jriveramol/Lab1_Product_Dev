# src/evaluate.py
import pandas as pd
import numpy as np
import joblib
import json
import yaml
import sys
import os
from sklearn.metrics import mean_squared_error, r2_score

def load_data(input_file):
    '''Carga la data limpia'''
    print(f'Cargando archivo {input_file}')
    try:
        data = pd.read_csv(input_file)
        print(f'Archivo cargado con éxito.')
    except:
        print(f'Error al cargar archivo.')
    print(100*'-', '\n')
    return data
    
def load_model(model_dir):
    '''Carga de modelos entrenados'''
    models_dict = dict()
    models = os.listdir(model_dir)
    for model in models:
        if 'pkl' in model:
            print(f'Cargando modelo {model}')
            name = model[:-4]
            models_dict[name] = joblib.load(f'{model_dir}/{model}')
    print(100*'-', '\n')
    return models_dict

def evaluate(input_file, model_dir, metrics_file, params_file):
    data = load_data(input_file)
    models = load_model(model_dir)
    # Leer los hiperparámetros
    with open(params_file) as f:
        params = yaml.safe_load(f)
        
    target = params['preprocessing']['target']
    X_test = data.drop(columns=target)
    y_test = data[target]
    
    metrics=dict()
    for model_name, model in models.items():
        print(model_name)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f'{rmse =: ,.2f}')
        print(f'{r2 =: ,.2f}')
        metrics[model_name] = {'RMSE': rmse, 'R^2 Score': r2}
        importance_features = model.coef_ if model_name=='LinearRegression' else model.feature_importances_
        
        print('Importancia de variables:')
        for feature, importance in zip(X_test.columns, importance_features):
            print(f"{feature}:{importance}")
        champion = min(metrics, key=lambda x: metrics[x]['RMSE'])
        print(100*'-')

    print(f"\nEl mejor modelo es {champion} con un RMSE de {metrics[champion]['RMSE']:,.2f}")
    print()

    # Guardar métricas en un archivo JSON
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en {metrics_file}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo del modelo, archivo de métricas, archivo de parámetros
    input_file = sys.argv[1]
    model_dir = sys.argv[2]
    metrics_file = sys.argv[3]
    params_file = sys.argv[4]

    evaluate(input_file, model_dir, metrics_file, params_file)
