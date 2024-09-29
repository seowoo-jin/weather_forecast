
from enums.enums import Data, Date
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
import warnings
import numpy as np

# FutureWarning 무시
warnings.filterwarnings("ignore", category=FutureWarning)

def monthlyMaePlot(data):
    result_pred = data[Data.PREDICTED_OUTPUT_DATA]
    true_values = data[Data.TEST_OUTPUT_DATA]
    date_df = data[Date.DATE]
    columns = data[Data.TEST_OUTPUT_DATA].columns

    if isinstance(result_pred, np.ndarray):
        result_pred = pd.DataFrame(result_pred, columns=columns)

    result_pred.columns = [f"result_{col}" for col in result_pred.columns]
    true_values.columns = [f"true_{col}" for col in true_values.columns]

    result_pred = result_pred.reset_index(drop=True)
    true_values = true_values.reset_index(drop=True)

    pred_result = pd.concat([date_df, result_pred, true_values], axis=1)

    pred_result['date'] = pd.to_datetime(pred_result['date'])
    pred_result['month'] = pred_result['date'].dt.month

    monthly_data = {}
    # 월별로 데이터를 분리하여 저장
    for month, group in pred_result.groupby(pred_result['month']):

        predDataSet = group.iloc[:, [1, 2, 3]]  # Prediction columns
        realDataSet = group.iloc[:, [4, 5, 6]]  # Actual values columns

        # Calculate mae for each column
        mae_values = {}
        for pred_col, real_col in zip(predDataSet.columns, realDataSet.columns):
            mae_value = mean_absolute_error(predDataSet[pred_col], realDataSet[real_col])
            mae_values[pred_col] = mae_value

        for col, value in mae_values.items():
            if 'temperature_2m_max' in col:
                temp_max_mae = value
            if 'temperature_2m_min' in col:
                temp_min_mae = value
            if 'speed_10m_max' in col:
                spped_mae = value

        monthly_data[str(month)] = {
            'pred_data': predDataSet,
            'real_data': realDataSet,
            'temp_min_mae': temp_min_mae,
            'temp_max_mae': temp_max_mae,
            'spped_mae': spped_mae,
            'overall_mae': (temp_min_mae + temp_max_mae + spped_mae) / 3
        }
    ## 월별 mae 나타내기
    import seaborn as sns
    months = list(monthly_data.keys())
    temp_min_maes = [monthly_data[month]['temp_min_mae'] for month in months]
    temp_max_maes = [monthly_data[month]['temp_max_mae'] for month in months]
    spped_maes = [monthly_data[month]['spped_mae'] for month in months]
    overall_mae = [monthly_data[month]['overall_mae'] for month in months]

    mae_df = pd.DataFrame({
        'Month': months,
        'Temp Min mae': temp_min_maes,
        'Temp Max mae': temp_max_maes,
        'Speed mae': spped_maes,
        'Overall mae': overall_mae
    })
    months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=mae_df, x='Month', y='Temp Min mae', label='Temp Min mae', marker='o')
    sns.lineplot(data=mae_df, x='Month', y='Temp Max mae', label='Temp Max mae', marker='o')
    sns.lineplot(data=mae_df, x='Month', y='Speed mae', label='Speed mae', marker='o')
    sns.lineplot(data=mae_df, x='Month', y='Overall mae', label='Overall mae', marker='o')
    plt.xticks(ticks=range(0, 12), labels=months)
    plt.title('Monthly mae Values')
    plt.xlabel('Month')
    plt.ylabel('mae')
    plt.legend()
    # plt.grid(True)
    plt.show()