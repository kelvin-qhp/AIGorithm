import pandas as pd

class ExcelUtil:
    def read(fileName,sheetName=None):
        df = pd.read_excel(fileName,sheet_name =sheetName)
        print(f'Success to read file：{fileName} size:{df.shape}')
        return df
    def readCsv(fileName,sheetName=None):
        df = pd.read_csv(fileName)
        print(f'Success to read file：{fileName} size:{df.shape}')
        return df