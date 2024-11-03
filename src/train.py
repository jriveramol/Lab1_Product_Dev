# src/train.py
import pandas as pd
import joblib
import sys
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

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
    
def train(input_file, model_dir, params_file):
    '''Entrenamiento y optimización de modelos'''
    data = load_data(input_file)
    
    # Leer los hiperparámetros
    with open(params_file) as f:
        params = yaml.safe_load(f)
        
    target = params['preprocessing']['target']
    X_train = data.drop(columns=target)
    y_train = data[target]
        
    # Modelos a usar
    models = {'LinearRegression': LinearRegression(),
              'RandomForest': RandomForestRegressor(random_state=params['train']['random_state']),
              'GradientBoosting': GradientBoostingRegressor(random_state=params['train']['random_state'])
             }
    
    # Parámetros de optimización
    parameters = params['train']['params_optm']
    
    # Entrenar modelos
    best_models = {}
    for name, model in models.items():
        search = GridSearchCV(model,
                              parameters[name],
                              scoring=params['train']['scoring'],
                              cv=params['train']['cv'],
                              n_jobs=-1
                             )
        search.fit(X_train, y_train)
        best_models[name] = {'model':search.best_estimator_, 'mse':float(-search.best_score_)}

    # Mostrar los mejores modelos y sus hiperparámetros
    print('Mejores modelos:\n')
    for model_name, best_model in best_models.items():
        print(f'    * {model_name}: {best_model}')
        # Guardar el modelo entrenado
        model_file = f'{model_dir}/{model_name}.pkl'
        joblib.dump(best_model['model'], model_file)
        print(f'    Modelo entrenado y guardado en {model_file}')


if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo del modelo, archivo de hiperparámetros
    input_file = sys.argv[1]
    model_dir = sys.argv[2]
    params_file = sys.argv[3]

    train(input_file, model_dir, params_file)
