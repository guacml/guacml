import numpy as np
import pandas as pd
from guacml.enums import ColType
from dateutil import parser as date_parser


class ColumnAnalyzer:
    @staticmethod
    def analyze(dataframe):
        """Creates metadata about the columns and might change object columns
        to stricter types if doing so is safe.

        """
        col_data = []
        for col in dataframe.columns:
            col_info = ColumnAnalyzer.analyze_col(dataframe, col)
            if not isinstance(col_info, dict):
                raise Exception('Error analyzing col {0}'.format(col))
            col_data.append(col_info)

        # for display to the user
        info_cols = ['col_name', 'type', 'n_unique', 'n_unique_%', 'n_na_%', 'n_blank_%', 'example']
        col_infos = pd.DataFrame(col_data, columns=info_cols)

        # for processing
        meta_cols = ['col_name', 'type', 'derived_from', 'n_unique', 'n_na', 'n_blank']
        col_metadata = pd.DataFrame(col_data, columns=meta_cols)

        return col_infos, col_metadata

    @staticmethod
    def analyze_col(dataframe, col_name):
        col = dataframe[col_name]
        n_unique = col.nunique()
        n_unique_pct = n_unique * 100 / len(col)
        not_null = col[col.notnull()]
        n_not_null = col.isnull().sum()

        col_info = {
            'col_name': col_name,
            'n_unique': n_unique,
            'n_unique_%': round(n_unique_pct),
            'n_na': n_not_null,
            'n_na_%': round(n_not_null * 100 / len(col)),
            'n_blank': 0,
            'n_blank_%': 0,
            'example': not_null.iloc[0]
        }

        if col.dtype == 'int64':
            return ColumnAnalyzer.analyze_int_col(col, n_unique, n_unique_pct, col_info)

        if col.dtype == 'float64':
            return ColumnAnalyzer.analyze_float_col(col_info)

        if col.dtype == 'datetime64[ns]':
            return ColumnAnalyzer.analyze_date_col(col_info)

        if col.dtype == 'object':
            return ColumnAnalyzer.analyze_object_col(
                dataframe, col_name, not_null, n_unique_pct, col_info
            )

    @staticmethod
    def analyze_int_col(col, n_unique, n_unique_pct, col_info):
        if col.min() == 0 and col.max() == 1:
            col_info['type'] = ColType.BINARY
            return col_info

            # all values from 0/1 until n_unique are present -> IDs or encoded categories
        if (col.min() == 0 or col.min() == 1) and (col.max() - col.min() == n_unique - 1):
            if n_unique_pct == 100:
                col_info['type'] = ColType.INT_ENCODING
                return col_info

            col_info['type'] = ColType.CATEGORICAL
            return col_info

        col_info['type'] = ColType.ORDINAL
        return col_info

    @staticmethod
    def analyze_float_col(col_info):
        col_info['type'] = ColType.NUMERIC
        return col_info

    @staticmethod
    def analyze_date_col(col_info):
        col_info['type'] = ColType.DATETIME
        return col_info

    @staticmethod
    def analyze_object_col(dataframe, col_name, not_null, n_unique_pct, col_info):
        col = dataframe[col_name]
        is_str = (not_null.values.dtype == np.str) or \
                 (isinstance(not_null.iloc[0], str) and isinstance(not_null.iloc[-1], str))
        if is_str:
            n_blank = (not_null == '').sum()
            col_info['n_blank'] = n_blank
            col_info['n_blank_%'] = round(n_blank * 100 / len(col))

            first = not_null.iloc[0]
            if first.isdecimal():
                try:
                    dataframe[col_name] = col.astype(int)
                    return ColumnAnalyzer.analyze_int_col(col, n_unique_pct, col_info)
                except ValueError:
                    pass
            elif first.isnumeric():
                try:
                    dataframe[col_name] = col.astype(float)
                    return ColumnAnalyzer.analyze_float_col(col_info)
                except ValueError:
                    pass
            else:
                try:
                    date_parser.parse(first)
                    dataframe[col_name] = pd.to_datetime(col)
                    return ColumnAnalyzer.analyze_date_col(col_info)
                except ValueError:
                    pass

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

            col_info['type'] = ColType.CATEGORICAL
            return col_info

        if isinstance(not_null.iloc[0], list) and isinstance(not_null.iloc[-1], list):
            col_info['type'] = ColType.LIST
            return col_info

        col_info['type'] = ColType.UNKNOWN
        return col_info
