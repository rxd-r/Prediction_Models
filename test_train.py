import csv
import pandas as pd

class Test_Train:
    def __init__(self):
        pass
    @staticmethod
    def cleaner():
        df = pd.read_csv('test_data.csv')
        for x in df.index:
            if df.loc[x, 'result'].strip("[']") == 'win':
                df.loc[x, 'result'] = 1
            else:
                df.loc[x, 'result'] = 0
        print(df)
        df.dropna()
        df.to_csv('test_data2.csv', encoding='utf-8', index=False)

    @staticmethod
    def test_train_split():
        pass