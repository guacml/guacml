import pandas as pd
import numpy as np
import datetime

from guacml.enums import ColType


class ColumnAnalyzer:
    """
    Creates metadata for a DataFrame.

    The `ColType` of the column has subtle differences to numpy types. Columns of
    type `ColType.BINARY` or `ColType.ORDINAL` can contain nulls and will be stored
    as numpy float columns.
    """
    def __init__(self, logger, type_check_samples=None):
        self.type_check_samples = type_check_samples
        self.logger = logger

    def analyze(self, df):
        """Creates metadata about the columns and might change object columns
        to stricter types if doing so is safe.

        """
        col_data = []
        for col in df.columns:
            ci = self.analyze_col(df, col)
            self.logger.info('Analyzed column %s as %s', col, ci['type'])
            col_data.append(ci)

        columns = ['col_name', 'type', 'derived_from', 'n_unique', 'n_na', 'n_blank', 'example']
        col_metadata = pd.DataFrame(col_data, columns=columns)
        return col_metadata.set_index('col_name')

    def analyze_col(self, df, col_name):
        col = df[col_name]
        n_unique = col.nunique()
        n_unique_pct = n_unique * 100 / len(col)
        not_null = col[col.notnull()]
        n_null = col.isnull().sum()

        if not_null.shape[0] > 0:
            example = not_null.iloc[0]
        else:
            example = None

        # ToDo: Rerun analysis when data type of column gets converted
        col_info = {
            'col_name': col_name,
            'n_unique': n_unique,
            'n_na': n_null,
            'n_blank': 0,
            'example': example
        }

        if col.dtype.kind == 'i':
            return self.analyze_int_col(df, col_name, n_unique, n_unique_pct, col_info)

        if col.dtype.kind == 'f':
            if (np.mod(not_null, 1) != 0).any():
                return self.analyze_float_col(col_info)
            else:
                if n_null == 0:
                    df[col_name] = col.astype(int)
                return self.analyze_int_col(df, col_name, n_unique, n_unique_pct, col_info)

        if col.dtype == 'datetime64[ns]':
            return self.analyze_date_col(col_info)

        if col.dtype == 'object':
            return self.analyze_object_col(df, col_name, not_null, n_null, n_unique, n_unique_pct,
                                           col_info)

        if col.dtype == 'bool':
            return self.analyze_boolean_col(col_info)

        raise Exception("Cannot analyze column '{}' of dtype '{}'".format(col_name, col.dtype))

    @staticmethod
    def analyze_int_col(df, col_name, n_unique, n_unique_pct, col_info):
        col = df[col_name]
        if col.min() == 0 and col.max() == 1:
            col_info['type'] = ColType.BINARY
            return col_info

        if len(col.name) >= 2 and col.name[-2:].lower() == 'id':
            col_info['type'] = ColType.INT_ENCODING
            return col_info

            # all values from 0/1 until n_unique are present -> IDs or encoded categories
        if (col.min() == 0 or col.min() == 1) and (col.max() - col.min() == n_unique - 1):
            if n_unique_pct == 100:
                col_info['type'] = ColType.INT_ENCODING
                return col_info
            else:

                col_info['type'] = ColType.CATEGORICAL
                df[col_name] = col.astype('str')
                return col_info

        col_info['type'] = ColType.ORDINAL
        return col_info

    @staticmethod
    def analyze_boolean_col(col_info):
        col_info['type'] = ColType.BINARY
        return col_info

    @staticmethod
    def analyze_float_col(col_info):
        col_info['type'] = ColType.NUMERIC
        return col_info

    @staticmethod
    def analyze_date_col(col_info):
        col_info['type'] = ColType.DATETIME
        return col_info

    def analyze_object_col(self, df, col_name, not_null, n_null, n_unique, n_unique_pct, col_info):
        """
        Checks the types of the referenced Python objects and calls the appropriate next step.
        """
        if self.type_check_samples is None:
            to_check = not_null.copy()
        else:
            to_check = not_null.sample(min(self.type_check_samples, len(not_null)))

        found_types = set()
        while len(to_check) > 0:
            current_type = type(to_check.iloc[0])
            found_types.add(current_type)
            to_check = to_check[to_check.apply(lambda x: type(x) != current_type)]

        col = df[col_name]
        date_types = {datetime.datetime, pd._libs.tslib.Timestamp, np.datetime64}
        if found_types.issubset(date_types):
            df[col_name] = pd.to_datetime(col)
            return self.analyze_date_col(col_info)

        if found_types == {bool}:
            df[col_name] = col.astype(bool)
            return self.analyze_boolean_col(col_info)

        if found_types == {bool, int}:
            df[col_name] = col.astype(int)
            return self.analyze_int_col(df, col_name, n_unique, n_unique_pct, col_info)

        if found_types.issubset({bool, int, float}):
            df[col_name] = col.astype(float)
            return self.analyze_float_col(col_info)

        if found_types.issubset({bool, int, float, str}.union(date_types)):
            return self.analyze_string_col(df, col_name, not_null, n_unique, n_unique_pct, col_info)
        else:
            col_info['type'] = ColType.UNKNOWN
            return col_info

    def analyze_string_col(self, df, col_name, not_null, n_unique, n_unique_pct, col_info):
        """
        Tries to parse the strings to stricter types or categorize the strings
        """
        col = df[col_name]
        n_blank = (not_null == '').sum()
        col_info['n_blank'] = n_blank

        if self.type_check_samples is None:
            to_check = not_null.copy()
        else:
            to_check = not_null.sample(min(self.type_check_samples, len(not_null)))

        if to_check.str.isdecimal().all():
            try:
                df[col_name] = col.astype(int)
                return self.analyze_int_col(df, col_name, n_unique, n_unique_pct, col_info)
            except ValueError:
                pass
        elif to_check.str.isnumeric().all():
            try:
                df[col_name] = col.astype(float)
                return self.analyze_float_col(col_info)
            except ValueError:
                pass
        else:
            try:
                df[col_name] = pd.to_datetime(col)
                return self.analyze_date_col(col_info)
            except (ValueError, TypeError):
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
        else:
            col_info['type'] = ColType.CATEGORICAL
            df[col_name] = col.astype('str')
            return col_info
