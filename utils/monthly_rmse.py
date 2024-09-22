
from enums.enums import Data, Date
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
import numpy as np

# FutureWarning 무시
warnings.filterwarnings("ignore", category=FutureWarning)

def monthlyRmsePlot(data):
    result_pred = data[Data.PREDICTED_OUTPUT_DATA.value]
    true_values = data[Data.TEST_OUTPUT_DATA.value]
    date_df = data[Date.DATE.value]
    columns = data[Data.TEST_OUTPUT_DATA.value].columns

    if isinstance(result_pred, np.ndarray):
        result_pred = pd.DataFrame(result_pred, columns=columns)


    result_pred.columns = [f"result_{col}" for col in columns]
    true_values.columns = [f"true_{col}" for col in columns]

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

        # Calculate RMSE for each column
        rmse_values = {}
        for pred_col, real_col in zip(predDataSet.columns, realDataSet.columns):
            rmse_value = mean_squared_error(predDataSet[pred_col], realDataSet[real_col], squared=False)
            rmse_values[pred_col] = rmse_value

        for col, value in rmse_values.items():
            if 'temperature_2m_max' in col:
                temp_max_rmse = value
            if 'temperature_2m_min' in col:
                temp_min_rmse = value
            if 'speed_10m_max' in col:
                spped_rmse = value

        monthly_data[str(month)] = {
            'pred_data': predDataSet,
            'real_data': realDataSet,
            'temp_min_rmse': temp_min_rmse,
            'temp_max_rmse': temp_max_rmse,
            'spped_rmse': spped_rmse,
            'overall_rmse': (temp_min_rmse + temp_max_rmse + spped_rmse) / 3
        }
    ## 월별 RMSE 나타내기
    import seaborn as sns
    months = list(monthly_data.keys())
    temp_min_rmses = [monthly_data[month]['temp_min_rmse'] for month in months]
    temp_max_rmses = [monthly_data[month]['temp_max_rmse'] for month in months]
    spped_rmses = [monthly_data[month]['spped_rmse'] for month in months]
    overall_rmse = [monthly_data[month]['overall_rmse'] for month in months]

    rmse_df = pd.DataFrame({
        'Month': months,
        'Temp Min RMSE': temp_min_rmses,
        'Temp Max RMSE': temp_max_rmses,
        'Speed RMSE': spped_rmses,
        'Overall RMSE': overall_rmse
    })
    months = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul']

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=rmse_df, x='Month', y='Temp Min RMSE', label='Temp Min RMSE', marker='o')
    sns.lineplot(data=rmse_df, x='Month', y='Temp Max RMSE', label='Temp Max RMSE', marker='o')
    sns.lineplot(data=rmse_df, x='Month', y='Speed RMSE', label='Speed RMSE', marker='o')
    sns.lineplot(data=rmse_df, x='Month', y='Overall RMSE', label='Overall RMSE', marker='o')
    plt.xticks(ticks=range(0, 12), labels=months)
    plt.title('Monthly RMSE Values')
    plt.xlabel('Month')
    plt.ylabel('RMSE')
    plt.legend()
    # plt.grid(True)
    plt.show()