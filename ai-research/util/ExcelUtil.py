import pandas as pd

class ExcelUtil:
    def read(self,fileName,sheetName=None):
        return pd.read_csv(fileName,sheetName)