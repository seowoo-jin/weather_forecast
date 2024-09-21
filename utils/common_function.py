import pandas as pd


def splitData(X, y, test_size: int):
    # 훈련 데이터와 테스트 데이터를 나누기
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # DataFrame으로 변환
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    y_train_df = pd.DataFrame(y_train, columns=y.columns)  # y가 Series인 경우 y.name으로 열 이름 지정
    y_test_df = pd.DataFrame(y_test, columns=y.columns)

    return X_train_df, X_test_df, y_train_df, y_test_df
