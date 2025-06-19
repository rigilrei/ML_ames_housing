import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


def load_housing_data(file_path='AmesHousing.csv'):
    print("Загрузка и подготовка данных...")
    try:
        housing_data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{file_path}' не найден.")

    # Удаляем ненужные колонки
    housing_data.drop(['Order', 'PID'], axis=1, inplace=True, errors='ignore')

    # Разделяем числовые и категориальные признаки
    numeric_features = housing_data.select_dtypes(include=np.number).columns
    categorical_features = housing_data.select_dtypes(exclude=np.number).columns

    # Обработка пропущенных значений
    housing_data[numeric_features] = housing_data[numeric_features].fillna(
        housing_data[numeric_features].median())
    housing_data[categorical_features] = housing_data[categorical_features].fillna('Unknown')

    # Преобразование категориальных признаков
    housing_data = pd.get_dummies(housing_data,
                                  columns=categorical_features,
                                  drop_first=True)

    # Масштабирование числовых признаков
    print("Масштабирование числовых признаков...")
    if len(numeric_features) > 0:
        scaler = StandardScaler()
        housing_data[numeric_features] = scaler.fit_transform(housing_data[numeric_features])

    return housing_data


def visualize_housing_prices(data):
    if 'SalePrice' not in data.columns:
        raise ValueError("Отсутствует целевая переменная 'SalePrice'.")

    print("Создание 3D визуализации...")
    try:
        features = data.drop('SalePrice', axis=1).select_dtypes(include=np.number)

        # Уменьшение размерности
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)

        # Создание 3D графика
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        price_plot = ax.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            data['SalePrice'],
            c=data['SalePrice'],
            cmap='plasma',
            alpha=0.7,
            s=50
        )

        ax.set_xlabel('Первая главная компонента')
        ax.set_ylabel('Вторая главная компонента')
        ax.set_zlabel('Цена продажи ($)')
        plt.title('3D визуализация данных о недвижимости')
        fig.colorbar(price_plot, ax=ax, label='Цена продажи ($)')

        plt.savefig('housing_3d_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Ошибка при визуализации: {str(e)}")


def optimize_lasso_model(data):
    if 'SalePrice' not in data.columns:
        raise ValueError("Отсутствует целевая переменная 'SalePrice'.")

    print("\nПодготовка данных для моделирования...")
    X = data.drop('SalePrice', axis=1).select_dtypes(include=np.number)
    y = data['SalePrice']

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Поиск оптимального параметра регуляризации
    alpha_values = np.logspace(-3, 3, 20)
    performance_metrics = []

    print("Поиск оптимальных параметров модели...")
    for alpha in alpha_values:
        try:
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            performance_metrics.append(rmse)
            print(f"α = {alpha:.5f} \t RMSE = {rmse:.2f}")
        except Exception as e:
            print(f"Ошибка при α={alpha}: {str(e)}")
            performance_metrics.append(np.nan)

    if all(np.isnan(performance_metrics)):
        raise ValueError("Все модели завершились с ошибкой!")

    # Находим лучшую модель
    optimal_alpha = alpha_values[np.nanargmin(performance_metrics)]
    best_rmse = np.nanmin(performance_metrics)

    print(f"\nОптимальный параметр регуляризации: α = {optimal_alpha:.5f}")
    print(f"Лучшая RMSE: {best_rmse:.2f}")

    # Визуализация результатов
    plt.figure(figsize=(12, 7))
    plt.semilogx(alpha_values, performance_metrics, 'o-')
    plt.xlabel('Коэффициент регуляризации (α)')
    plt.ylabel('RMSE')
    plt.title('Зависимость качества модели от параметра регуляризации')
    plt.grid(True, which="both", ls="--")
    plt.axvline(optimal_alpha, color='r', linestyle='--',
                label=f'Оптимальный α = {optimal_alpha:.3f}')
    plt.legend()
    plt.savefig('lasso_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return X_train, X_test, y_train, y_test, optimal_alpha


def analyze_model_features(X_train, y_train, optimal_alpha):
    print("\nАнализ влияния признаков...")
    try:
        # Обучаем модель с оптимальным параметром
        final_model = Lasso(alpha=optimal_alpha, max_iter=10000)
        final_model.fit(X_train, y_train)

        # Создаем DataFrame с коэффициентами
        feature_importance = pd.DataFrame({
            'Признак': X_train.columns,
            'Влияние': final_model.coef_
        }).sort_values('Влияние', key=abs, ascending=False)

        # Выводим топ-10 важных признаков
        top_features = feature_importance.head(10)
        print("\nТоп-10 наиболее влиятельных признаков:")
        print(top_features)

        # Визуализация
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Влияние', y='Признак',
                    data=top_features,
                    palette='viridis')
        plt.title('10 наиболее значимых признаков для цены недвижимости')
        plt.tight_layout()
        plt.savefig('feature_significance.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Ошибка при анализе признаков: {str(e)}")


if __name__ == "__main__":
    try:
        # Основной процесс анализа
        housing_df = load_housing_data()
        print(f"\nУспешно загружено {housing_df.shape[0]} записей с {housing_df.shape[1]} признаками")

        visualize_housing_prices(housing_df)

        X_tr, X_te, y_tr, y_te, best_alpha = optimize_lasso_model(housing_df)

        analyze_model_features(X_tr, y_tr, best_alpha)

        print("\nАнализ данных о недвижимости успешно завершен!")
    except Exception as e:
        print(f"\nПроизошла ошибка: {str(e)}")