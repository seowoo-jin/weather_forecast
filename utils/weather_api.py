import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from IPython.display import display, HTML

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

import time

# https://open-meteo.com/en/docs/historical-weather-api#latitude=41.5575&longitude=-89.4609&start_date=2010-01-01&end_date=2019-12-31&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,rain,snowfall,snow_depth,pressure_msl,surface_pressure,wind_speed_10m,wind_speed_100m,wind_direction_10m,wind_direction_100m&timezone=GMT
# https://open-meteo.com/en/docs/historical-weather-api
### input parameter description
# latitude
# longitude	Floating point	Yes		Geographical WGS84 coordinates of the location. Multiple coordinates can be comma separated. E.g. &latitude=52.52,48.85&longitude=13.41,2.35. To return data for multiple locations the JSON output changes to a list of structures. CSV and XLSX formats add a column location_id.
# elevation	Floating point	No		The elevation used for statistical downscaling. Per default, a 90 meter digital elevation model is used. You can manually set the elevation to correctly match mountain peaks. If &elevation=nan is specified, downscaling will be disabled and the API uses the average grid-cell height. For multiple locations, elevation can also be comma separated.
# start_date
# end_date	String (yyyy-mm-dd)	Yes		The time interval to get weather data. A day must be specified as an ISO8601 date (e.g. 2022-12-31).
# hourly	String array	No		A list of weather variables which should be returned. Values can be comma separated, or multiple &hourly= parameter in the URL can be used.
# daily	String array	No		A list of daily weather variable aggregations which should be returned. Values can be comma separated, or multiple &daily= parameter in the URL can be used. If daily weather variables are specified, parameter timezone is required.
# temperature_unit	String	No	celsius	If fahrenheit is set, all temperature values are converted to Fahrenheit.
# wind_speed_unit	String	No	kmh	Other wind speed speed units: ms, mph and kn
# precipitation_unit	String	No	mm	Other precipitation amount units: inch
# timeformat	String	No	iso8601	If format unixtime is selected, all time values are returned in UNIX epoch time in seconds. Please note that all time is then in GMT+0! For daily values with unix timestamp, please apply utc_offset_seconds again to get the correct date.
# timezone	String	No	GMT	If timezone is set, all timestamps are returned as local-time and data is returned starting at 00:00 local-time. Any time zone name from the time zone database is supported If auto is set as a time zone, the coordinates will be automatically resolved to the local time zone. For multiple coordinates, a comma separated list of timezones can be specified.
# cell_selection	String	No	land	Set a preference how grid-cells are selected. The default land finds a suitable grid-cell on land with similar elevation to the requested coordinates using a 90-meter digital elevation model. sea prefers grid-cells on sea. nearest selects the nearest possible grid-cell.
# apikey	String	No		Only required to commercial use to access reserved API resources for customers. The server URL requires the prefix customer-. See pricing for more information.


### output parameter description
# Variable	Unit	Description
# weather_code	WMO code	The most severe weather condition on a given day
# temperature_2m_max
# temperature_2m_min	°C (°F)	Maximum and minimum daily air temperature at 2 meters above ground
# apparent_temperature_max
# apparent_temperature_min	°C (°F)	Maximum and minimum daily apparent temperature
# precipitation_sum	mm	Sum of daily precipitation (including rain, showers and snowfall)
# rain_sum	mm	Sum of daily rain
# snowfall_sum	cm	Sum of daily snowfall
# precipitation_hours	hours	The number of hours with rain
# sunrise
# sunset	iso8601	Sun rise and set times
# sunshine_duration	seconds	The number of seconds of sunshine per day is determined by calculating direct normalized irradiance exceeding 120 W/m², following the WMO definition. Sunshine duration will consistently be less than daylight duration due to dawn and dusk.
# daylight_duration	seconds	Number of seconds of daylight per day
# wind_speed_10m_max
# wind_gusts_10m_max	km/h (mph, m/s, knots)	Maximum wind speed and gusts on a day
# wind_direction_10m_dominant	°	Dominant wind direction
# shortwave_radiation_sum	MJ/m²	The sum of solar radiaion on a given day in Megajoules
# et0_fao_evapotranspiration	mm	Daily sum of ET₀ Reference Evapotranspiration of a well watered grass field
###
def get_weather_data_with_retry(x, y, start_date, end_date):
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": x,
        "longitude": y,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth", "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m"],
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"],
        "timezone": "GMT"
    }
    while True:
        try:
            responses = openmeteo.weather_api(url, params=params)
            print("요청 성공")
            return responses  # 성공 시 데이터를 반환하고 루프 종료
        except:
            print("요청 실패: 10 초 후 재시도합니다.")
            time.sleep(10)  # 지정된 시간 동안 대기 후 다시 시도

class WeatherApi:

    async def get_weather_data(self, x, y, start_date, end_date):

        responses = get_weather_data_with_retry(x, y, start_date, end_date)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        # Process daily data. The order of variables needs to be the same as requested.
        daily = response.Daily()
        daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
        daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()
        daily_apparent_temperature_max = daily.Variables(3).ValuesAsNumpy()
        daily_apparent_temperature_min = daily.Variables(4).ValuesAsNumpy()
        daily_apparent_temperature_mean = daily.Variables(5).ValuesAsNumpy()
        daily_sunrise = daily.Variables(6).ValuesAsNumpy()
        daily_sunset = daily.Variables(7).ValuesAsNumpy()
        daily_daylight_duration = daily.Variables(8).ValuesAsNumpy()
        daily_sunshine_duration = daily.Variables(9).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(10).ValuesAsNumpy()
        daily_rain_sum = daily.Variables(11).ValuesAsNumpy()
        daily_snowfall_sum = daily.Variables(12).ValuesAsNumpy()
        daily_precipitation_hours = daily.Variables(13).ValuesAsNumpy()
        daily_wind_speed_10m_max = daily.Variables(14).ValuesAsNumpy()
        daily_wind_gusts_10m_max = daily.Variables(15).ValuesAsNumpy()
        daily_wind_direction_10m_dominant = daily.Variables(16).ValuesAsNumpy()
        daily_shortwave_radiation_sum = daily.Variables(17).ValuesAsNumpy()
        daily_et0_fao_evapotranspiration = daily.Variables(18).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )}
        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
        # daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
        # daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
        # daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
        # daily_data["sunrise"] = daily_sunrise
        # daily_data["sunset"] = daily_sunset
        daily_data["daylight_duration"] = daily_daylight_duration
        daily_data["sunshine_duration"] = daily_sunshine_duration
        # daily_data["precipitation_sum"] = daily_precipitation_sum
        # daily_data["rain_sum"] = daily_rain_sum
        # daily_data["snowfall_sum"] = daily_snowfall_sum
        # daily_data["precipitation_hours"] = daily_precipitation_hours
        daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
        daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
        daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
        # daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
        # daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration

        daily_dataframe = pd.DataFrame(data = daily_data)

        return daily_dataframe

    def get_weather_data_from_excel(self, fileName):
        ### temperature_2m_max	temperature_2m_min	temperature_2m_mean	daylight_duration	sunshine_duration	wind_speed_10m_max	wind_gusts_10m_max	wind_direction_10m_dominant
        file_path = f'data/{fileName}.xlsx'

        # 엑셀 파일을 읽고 각 시트를 데이터프레임으로 변환
        xlsx = pd.ExcelFile(file_path)
        sheets = xlsx.sheet_names  # 시트 이름 리스트

        # 각 시트를 데이터프레임으로 저장
        dfs = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}

        target_sheet = fileName

        if target_sheet in dfs:
            y = dfs[target_sheet]
            y = y.drop(columns=['sunshine_duration', 'wind_gusts_10m_max', 'wind_direction_10m_dominant', 'daylight_duration', 'temperature_2m_mean'])
            y.columns = [f"{col}_{target_sheet}" for col in y.columns]


        # 빈 데이터프레임 생성 (첫 번째 시트를 기준으로 초기화)
        X = pd.read_excel(file_path, sheet_name=sheets[0])

        # 첫 번째 시트의 컬럼 이름에 시트 이름을 추가
        X.columns = [f"{col}_{sheets[0]}" if col != 'date' else col for col in X.columns]

        # 나머지 시트를 date 컬럼 기준으로 병합
        for sheet in sheets[1:]:
            df = pd.read_excel(file_path, sheet_name=sheet)

            if(sheet != target_sheet):
                # 병합할 데이터프레임의 컬럼 이름에 시트 이름 추가
                df.columns = [f"{col}_{sheet}" if col != 'date' else col for col in df.columns]
                X = pd.merge(X, df, on='date', how='outer')

        # 하루 shift
        total_data = pd.concat([X.iloc[:-1, :], y.iloc[1:]], axis=1)
        total_data = total_data.dropna(axis=0)


        # 특정 단어 (예: 'temperature')가 포함된 컬럼명 추출
        filtered_columns = [col for col in total_data.columns if target_sheet in col]

        # 해당 컬럼들로만 구성된 DataFrame 생성
        recreated_y = total_data[filtered_columns]
        recreated_X = total_data.drop(filtered_columns, axis=1)

        return recreated_X, recreated_y
