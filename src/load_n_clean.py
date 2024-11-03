# src/load_n_clean.py
import pandas as pd
import yaml

def load_data(input_file):
    '''Carga la data'''
    print(f'Cargando archivo {input_file}')
    try:
        data = pd.read_csv(input_file)
        print(f'Archivo cargado con éxito.')
    except:
        print(f'Error al cargar archivo.')
    print(100*'-', '\n')
    return data

def null_data(data):
    '''Muestra la cantidad de valores nulos por columna'''
    missing_summary = data.isnull().sum()
    print("Valores faltantes por columna:")
    print(missing_summary)
    print(100*'-', '\n')
    return missing_summary

def data_imputation(data):
    '''Imputación de valores nulos con el valor más frecuente'''
    print('Imputación de datos nulos')
    for column in data.columns:
        if data[column].dtype == 'float64' or data[column].dtype == 'int64':
            median_value = data[column].median()
            data[column] = data[column].fillna(median_value)
        else:
            mode_value = data[column].mode()[0]
            data[column] = data[column].fillna(mode_value)
    print(100*'-', '\n')
    return data

def data_type_validation(data, params_file):
    '''Valida que las variables que deberían ser numéricas sí lo sean'''
    print('Validación de tipo de dato')
     # Leer los parámetros
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Separar características y variable objetivo
    num_feat = params['preprocessing']['numeric_features']
    tar_feat = params['preprocessing']['target']
    
    num_cols = num_feat + [tar_feat]
    
    print("Tipos de datos originales:")
    print(data.dtypes)
    print(num_cols)
    for column in num_cols:
        if column in data.columns and data[column].dtype != 'int64':
            data[column] = data[column].astype(int)
    print(100*'-', '\n')
    return data

def clean_data(input_file, output_file, params_file):
    '''Pipeline que aplica la carga de datos, muestra los nulos, los imputa y valida el tipo de dato'''
    data = load_data(input_file)
    null_data(data)
    data = data_imputation(data)
    data = data_type_validation(data, params_file)
    data.to_csv(output_file, index=False)
    print(f"Datos limpios guardados en {output_file}")

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    params_file = sys.argv[3]
    clean_data(input_file, output_file, params_file)
