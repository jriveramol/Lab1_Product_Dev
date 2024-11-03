# src/preprocess.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import yaml
import sys

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

def encode_data(data, categorical_columns, numerical_columns, target):
    '''Decodificación de variables categóricas y numéricas'''
    # Se crea un preprocesador de las columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns),
            ('num', StandardScaler(), numerical_columns)
        ])
    print("Aplicando preprocesador")
    # Se aplica el preprocesador
    processed_data = preprocessor.fit_transform(data)
    processed_df = pd.DataFrame(processed_data, columns=preprocessor.get_feature_names_out())
    processed_df = pd.concat([processed_df, data[target]], axis=1)
    print(100*'-', '\n')
    return processed_df
    
def data_division(data, target_column, test_size, random_state):
    '''División de dataset'''
    print("Dividiendo dataset")
    train_set, test_set = train_test_split(data, test_size=test_size, random_state=random_state)
    test_set, validation_set = train_test_split(test_set, test_size=0.5, random_state=random_state)
    
    print(f'Cant. registros set entrenamiento: {len(train_set)}')
    print(f'Cant. registros set prueba: {len(test_set)}')
    print(f'Cant. registros set validación: {len(validation_set)}')
    print(100*'-', '\n')
    return train_set, test_set, validation_set
    
def preprocess(input_file, output_train, output_test, output_val, params_file):
    '''Preprocesamiento de la data'''
    data = load_data(input_file)
    
    # Leer parámetros desde params.yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)
    
    # Leer columnas categóricas y numéricas desde params.yaml
    categorical_columns = params['preprocessing']['categorical_features']
    numerical_columns = params['preprocessing']['numeric_features']
    target = params['preprocessing']['target']
    test_size = params['preprocessing']['test_size']
    random_state = params['preprocessing']['random_state']
    
    data = encode_data(data, categorical_columns, numerical_columns, target)
    train_data, test_data, validation_data = data_division(data, target, test_size, random_state)
    
    train_data.to_csv(output_train, index=False)
    print(f"Datos de entrenamiento guardados en {output_train}")
    
    test_data.to_csv(output_test, index=False)
    print(f"Datos de prueba guardados en {output_test}")
    
    validation_data.to_csv(output_val, index=False)
    print(f"Datos de validación guardados en {output_val}")

if __name__ == "__main__":
    # Argumentos: archivo de entrada, archivo de salida, características y objetivo
    input_file = sys.argv[1]
    output_train = sys.argv[2]
    output_test = sys.argv[3]
    output_val = sys.argv[4]
    params_file = sys.argv[5]

    preprocess(input_file, output_train, output_test, output_val, params_file)
