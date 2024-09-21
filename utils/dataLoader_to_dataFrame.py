import pandas as pd

def change(x_columns, y_columns, loader):
    for inputs, labels in loader:
        inputs_numpy = inputs.numpy()
        labels_numpy = labels.numpy()

        # numpy 배열을 DataFrame으로 변환
        inputs_df = pd.DataFrame(inputs_numpy, columns=[f'{x_columns[i]}' for i in range(inputs_numpy.shape[1])])
        inputs_df_y = pd.DataFrame(labels_numpy, columns=[f'{y_columns[i]}' for i in range(labels_numpy.shape[1])])

    return pd.concat([inputs_df, inputs_df_y], axis=1)
