# 날씨 예측 모델
## 사용된 모델
### 기본 모델 트리
  - Decision Tree
  - Extra Tree
### 기본 모델 Layer
  - Multi Layer Perceptron
### 앙상블 모델
  - Gradient boosting
  - Random Forest
  - Xgboost
### 앙상블 기법
  - Voting
  - Staking
  - Bagging
~~~
 ┣ data
 ┃ ┣ Billings_MT.xlsx         * 몬타나주 기상데이터
 ┃ ┣ Swanton_OH.xlsx          * 오하이오주 기상데이터
 ┃ ┣ montana_data.ipynb       * 몬타나주 기상데이터 API
 ┃ ┗ ohio_data.ipynb          * 오하이오주 기상데이터 API
 ┣ enums
 ┃ ┗ enums.py                 * Enum
 ┣ result_model_fold          * 학습된 모델들 모음
 ┃ ┣ Billings_MT              * 몬타나주 학습 결과 모델 pkl 파일
 ┃ ┗ Swanton_OH               * 오하이오주 학습 결과 모델 pkl 파일
 ┣ training_result            * 모델 학습 시킨 기록 결과 모음 
 ┃ ┣ Billings_MT              * 몬타나주 모델 학습 결과 기록
 ┃ ┣ Swanton_OH               * 오하이오주 모델 학습 결과 기록
 ┣ utils
 ┃ ┣ __init__.py
 ┃ ┣ common_function.py          * 공통함수
 ┃ ┣ create_plot.py              * 그래프 생성
 ┃ ┣ dataLoader_to_dataFrame.py  * numpy to dataframe 함수
 ┃ ┣ get_models_from_result.py   * 학습 결과 모델 pkl 파일에서 rmse, plot 생성
 ┃ ┣ monthly_mae.py              * 월별 mae 값 비교
 ┃ ┣ monthly_rmse.py             * 월별 rmse 값 비교
 ┃ ┗ weather_api.py              * 날씨 몌ㅑ
 ┣ MLP.ipynb                     * MLP 모델 결과
 ┣ Begging.ipynb                 * Begging 모델 학습
 ┣ Decisiontree.ipynb            * 결정 트리 모델 결과
 ┣ ExtraTree.ipynb               * Extra 트리 모델 결과
 ┣ Note                          * 노트
 ┣ RandomForest.ipynb            * Random Forest 모델 결과
 ┣ Staking.ipynb                 * Staking 모델 학습
 ┣ Voting.ipynb                  * Voting 모델 학습
 ┣ Xgboost.ipynb                 * Xgboost 모델 결과
 ┣ compare_MT_model.ipynb        * 몬타나주의 모든 학습된 모델 성능 비교
 ┣ compare_OH_model.ipynb        * 오하이오주의 모든 학습된 모델 성능 비교
 ┣ requirements.txt
 ┣ training_base_model.ipynb     * 기본 모델 학습
 ┗ 논문.docx
~~~
## 데이터
### API URL
- https://open-meteo.com/en/docs/historical-weather-api
### 날씨 데이터 지역
- Ohio 외 주변 8지역
# ![image](https://github.com/user-attachments/assets/93e4492a-6815-44f9-8d43-98bb2eee558d)
- Montana 외 주변 8지역
# ![image](https://github.com/user-attachments/assets/c7f8b2b7-eb76-4dd5-be7d-357544e77b8b)


