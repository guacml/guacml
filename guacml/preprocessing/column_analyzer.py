import numpy as np
import pandas as pd
from enum import Enum


ColType = Enum('ColType', 'ID BINARY CATEGORICAL NUMERIC ORDINAL DATETIME TEXT WORDS LIST UNKNOWN')

class ColumnAnalyzer:
    def analyze(self, df):
        col_infos = []
        for col in df.columns:
            ci = self.analyze_col(df[col], col)
            col_infos.append(ci)
        col_infos = pd.DataFrame(col_infos,
                            columns=['col_name', 'type', 'n_unique', 'n_unique_%',
                                     'n_na_%', 'n_blank_%', 'example'])
        return col_infos

    def analyze_col(self, col, col_name):
        n_unique = col.nunique()
        n_unique_pct = n_unique * 100 / len(col)
        not_null = col[col.notnull()]

        col_info = {
            'col_name': col_name,
            'n_unique': n_unique,
            'n_unique_%': round(n_unique_pct),
            'n_na_%': round(col.isnull().sum() * 100 / len(col)),
            'n_blank_%': 0,
            'example': not_null.iloc[0]
        }

        if col.dtype == 'int64':
            if col.min() == 0 and col.max() == 1:
                col_info['type'] = 'ColType.BINARY'
                return col_info

            # all values from 0/1 until n_unique are present -> IDs or encoded categories
            if (col.min() == 0 or col.min() == 1) and (col.max() - col.min() == n_unique - 1):
                if n_unique_pct == 100:
                    col_info['type'] = ColType.ID
                    return col_info
                else:
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
            is_str = (not_null.values.dtype == np.str) or \
                     (type(not_null.iloc[0]) is str and type(not_null.iloc[-1]) is str)
            if is_str:
                col_info['n_blanks_%'] = round((not_null == '').sum() * 100 / len(col))
                word_counts = not_null.str.count('[ ,;]')
                mean_words = word_counts.mean()
                if mean_words >= 5:
                    col_info['type'] = ColType.TEXT
                    return col_info
                if mean_words > 1.2:
                    col_info['type'] = ColType.WORDS
                    return col_info
                if mean_words == 1 and n_unique_pct == 100:
                    col_info['type'] = ColType.ID
                    return col_info
                if mean_words == 1 and col.str.lower.isin('true', 'false', '0', '1'):
                    col_info['type'] = ColType.BINARY
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