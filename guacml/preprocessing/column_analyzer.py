import numpy as np
import pandas as pd
from enum import Enum


ColType = Enum('ColType', 'ID CATEGORICAL NUMERIC ORDINAL DATETIME TEXT WORDS LIST UNKNOWN')
Cardinality = Enum('Cardinality', 'LOW MEDIUM HIGH')


class ColumnAnalyzer:
    def analyze(self, df):
        col_infos = []
        for col in df.columns:
            ci = self.analyze_col(df[col], col)
            col_infos.append(ci)
        return pd.DataFrame(col_infos, columns=['col_name', 'type', 'contains_na', 'cardinality'])

    def analyze_col(self, col, col_name):
        n_unique = col.nunique()
        length = len(col)
        contains_na = (col.isnull().sum() > 0)
        col_info = {'col_name': col_name, 'contains_na': contains_na}

        if n_unique == length:
            col_info['type'] = ColType.ID
            col_info['cardinality'] = Cardinality.HIGH
            return col_info
        elif n_unique > 250:
            card = Cardinality.HIGH
        elif n_unique  > 25:
            card = Cardinality.MEDIUM
        else:
            card = Cardinality.LOW
        col_info['cardinality'] = card

        if col.dtype == 'int64':
            # all values from 0/1 until n_unique are present -> encoded categories
            if (col.min() == 0 or col.min() == 1):
                if col.max() - col.min() == n_unique - 1:
                    col_info['type'] = ColType.CATEGORICAL
                    return col_info
            col_info['type'] = ColType.ORDINAL
            return col_info

        if col.dtype == 'float64':
            col_info['type'] = ColType.NUMERIC
            return col_info
        if col.dtype == 'datetime64[ns]':
            col_info['type'] = ColType.DATETIME
            return col_info

        if col.dtype == 'object':
            not_null = col[col.notnull()]

            is_str = (not_null.values.dtype == np.str) or \
                     (type(not_null.iloc[0]) is str and type(not_null.iloc[-1]) is str)
            if is_str:
                n_blanks = (not_null == '').sum()
                if n_blanks > 0:
                    not_null = not_null.replace('', None)
                    print('Column {0}: Replaced {1} blank values with None.'.format(col_name, n_blanks))

                word_counts = not_null.str.count(' ')
                mean_words = word_counts.mean()
                if mean_words >= 5:
                    col_info['type'] = ColType.TEXT
                    return col_info
                if mean_words > 1.5:
                    col_info['type'] = ColType.WORDS
                    return col_info
                else:
                    col_info['type'] = ColType.CATEGORICAL
                    return col_info

            if type(not_null.iloc[0]) is list and type(not_null.iloc[-1]) is list:
                col_info['type'] = ColType.LIST
                return col_info
            else:
                col_info['type'] = ColType.UNKNOWN
                return col_info