import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

def import_dataset_from_file(path_to_file: str) -> pd.DataFrame:
    """
    Функция импортирования исходных данных.
    :param path_to_file: путь к загружаемому файлу;
    :return: структура данных.
    """
    dataset = pd.read_table(path_to_file, delim_whitespace=True, names=['x', 'y', 'z'])

    return dataset

def export_dataset_to_file(dataset: pd.DataFrame):
    """
    Функция экспортирования результата в файл result.txt.
    :param dataset: входная структура данных.
    """
    n, c = dataset.shape

    assert c == 3, 'Количество столбцов должно быть 3'
    assert n == 1196590, 'Количество строк должно быть 1196590'

    with open('Data\\Result.txt', 'w') as f:
        for i in range(n):
            f.write('%.2f %.2f %.5f\n' % (dataset.x[i], dataset.y[i], dataset.z[i]))

if __name__ == "__main__":
    # Вспомогательные данные, по которым производится моделирование
    map_1_dataset = import_dataset_from_file("Data\\Map_1.txt")
    map_2_dataset = import_dataset_from_file("Data\\Map_2.txt")
    map_3_dataset = import_dataset_from_file("Data\\Map_3.txt")
    map_4_dataset = import_dataset_from_file("Data\\Map_4.txt")
    map_5_dataset = import_dataset_from_file("Data\\Map_5.txt")

    # Данные, по которым необходимо смоделировать
    point_dataset = import_dataset_from_file("Data\\Point_dataset.txt")

    # Точки данных, в которые необходимо провести моделирование (сетка данных)
    point_grid = import_dataset_from_file("Data\\Result_schedule.txt")

    map_1_dataset = map_1_dataset.rename(columns={'z': 'Map_1'})

    # Блок вычислений
    merged_data = pd.merge(map_1_dataset, map_2_dataset, on=['x', 'y'], how='outer')
    merged_data = merged_data.rename(columns={'z': 'Map_2'})
    merged_data = pd.merge(merged_data, map_3_dataset, on=['x', 'y'], how='outer')
    merged_data = merged_data.rename(columns={'z': 'Map_3'})
    merged_data = pd.merge(merged_data, map_4_dataset, on=['x', 'y'], how='outer')
    merged_data = merged_data.rename(columns={'z': 'Map_4'})
    merged_data = pd.merge(merged_data, map_5_dataset, on=['x', 'y'], how='outer')
    merged_data = merged_data.rename(columns={'z': 'Map_5'})
    merged_data = merged_data.rename(columns={'x': 'X1'})
    merged_data = merged_data.rename(columns={'y': 'Y1'})

    # Выбираем признаки (Map_1, Map_2, Map_3, Map_4, Map_5, X, Y)
    features = ['Map_1', 'Map_2', 'Map_3', 'Map_4', 'Map_5', 'X1', 'Y1']

    # Создаем экземпляр KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)

    # Заполняем пропуски в признаках Map_1 до Map_5 на основе значений X и Y
    merged_data[features] = knn_imputer.fit_transform(merged_data[features])

    maps_data = merged_data[:]

    points_data = point_dataset[:]
    points_data = points_data.rename(columns={'x': 'X2'})
    points_data = points_data.rename(columns={'y': 'Y2'})
    points_data = points_data.rename(columns={'z': 'Z'})

    # Создаем модель KNN
    knn_model = KNeighborsRegressor(n_neighbors=2)

    # Обучаем модель на данных maps_data
    knn_model.fit(maps_data[['X1', 'Y1']], maps_data[['Map_1', 'Map_2', 'Map_3', 'Map_4', 'Map_5']])

    # Предсказание значений для новых точек (X2, Y2)
    predicted_values_knn = knn_model.predict(points_data[['X2', 'Y2']].values)

    # Добавление предсказанных значений в points_data
    points_data['Predicted_Map_1_KNN'] = predicted_values_knn[:, 0]
    points_data['Predicted_Map_2_KNN'] = predicted_values_knn[:, 1]
    points_data['Predicted_Map_3_KNN'] = predicted_values_knn[:, 2]
    points_data['Predicted_Map_4_KNN'] = predicted_values_knn[:, 3]
    points_data['Predicted_Map_5_KNN'] = predicted_values_knn[:, 4]

    points_data = points_data.rename(columns={'Predicted_Map_1_KNN': 'Map_1'})
    points_data = points_data.rename(columns={'Predicted_Map_2_KNN': 'Map_2'})
    points_data = points_data.rename(columns={'Predicted_Map_3_KNN': 'Map_3'})
    points_data = points_data.rename(columns={'Predicted_Map_4_KNN': 'Map_4'})
    points_data = points_data.rename(columns={'Predicted_Map_5_KNN': 'Map_5'})
    points_data = points_data.rename(columns={'X2': 'X1'})
    points_data = points_data.rename(columns={'Y2': 'Y1'})

    points = points_data[:]

    # Выбираем признаки (Map_1, Map_2, Map_3, Map_4, Map_5, X, Y) и целевую переменную (Z)
    features = ['Map_1', 'Map_2', 'Map_3', 'Map_4', 'Map_5', 'X1', 'Y1']
    target = 'Z'

    # Разделяем данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(points[features], points[target], test_size=0.2,
                                                        random_state=42)

    # Инициализируем модель Random Forest
    rf_model = RandomForestRegressor(n_estimators=1500, random_state=1)

    # Обучаем модель
    rf_model.fit(X_train, y_train)

    # Предсказываем значения для тестового набора
    y_pred_rf = rf_model.predict(X_test)


    result_data = point_grid[:]
    result_data = result_data.rename(columns={'x': 'X1'})
    result_data = result_data.rename(columns={'y': 'Y1'})
    result_data = result_data.rename(columns={'z': 'Z'})

    # Выбираем признаки для предсказания 'Z'
    features_for_prediction = ['Map_1', 'Map_2', 'Map_3', 'Map_4', 'Map_5', 'X1', 'Y1']

    # Применяем обученную модель для предсказания 'Z'
    merged_data['Predicted_Z'] = rf_model.predict(merged_data[features_for_prediction])


    # Создание словаря для замены NaN значений в 'Z'
    replacement_dict = merged_data.set_index(['X1', 'Y1'])['Predicted_Z'].to_dict()

    # Замена NaN значений в 'Z' в DataFrame 'result'
    result_data['Z'] = result_data.apply(lambda row: replacement_dict.get((row['X1'], row['Y1']), row['Z']), axis=1)

    # Выбираем признаки для заполнения (X, Y, Predicted_Z)
    features_to_impute = ['X1', 'Y1', 'Z']

    # Создаем копию данных для импутации
    imputation_data = result_data[features_to_impute].copy()

    # Инициализируем KNNImputer с количеством соседей, равным 3
    knn_imputer = KNNImputer(n_neighbors=3)

    # Заполняем NaN значения
    imputed_data = knn_imputer.fit_transform(imputation_data)

    # Создаем новый датафрейм с заполненными значениями
    imputed_values = pd.DataFrame(imputed_data, columns=features_to_impute)

    # Обновляем оригинальный датафрейм
    result_data[features_to_impute] = imputed_values
    result_data = result_data.rename(columns={'X1': 'x'})
    result_data = result_data.rename(columns={'Y1': 'y'})
    result_data = result_data.rename(columns={'Z': 'z'})
    print(result_data)

    # Экспорт данных в файл (смотри Readme.txt)
    # export_dataset_to_file(dataset=dataset)
    export_dataset_to_file(dataset=result_data)