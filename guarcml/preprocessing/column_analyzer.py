import guarcml.preprocessing.column_info

class ColumnAnalyzer:

    def __init__(self, df, col_name):
        self.df = df
        self.col_name = col_name

    def analyze(self):
        col = self.df[self.col_name]
        n_unique = col.nunique()
        length = len(col)
        if n_unique == length:
            return ColumnInfo(ColType.ID, Cardinality.HIGH)
        elif n_unique / length > 0.1:
            card = CARDINALITY.HIGH
        elif n_unique / length > 0.01:
            card = CARDINALITY.MEDIUM
        elif n_unique < :
            card = CARDINALITY.MEDIUM


        if col.dtype == 'int64':






        if col.dtype == 'float64':







        total_count = len(col)

        average_words = None
        try:
            unique_count = len(col.unique())
            if col.dtype == 'object':
                try:
                    word_count = self._word_count(col)
                    average_words = word_count.where(word_count > 0).mean()
                except AttributeError:
                    pass

        except TypeError:
            unique_count = len(col.apply(tuple).unique())


        # if col.dtype == 'object':
        #     # TODO distinguish betewen blank and missing
        #     null_count = col.apply(lambda x: len(x or '') == 0).sum()
        # else:
        null_count = col.isnull().sum()







        if col.dtype == 'datetime64[ns]':
            column_type = 'time'
        elif col.dtype == 'object' and type(col.values[0]) is list:
            column_type = 'list'
        elif average_words != None and average_words > 10:
            column_type = 'text'
        elif unique_count / float(total_count) > 0.95:
            column_type = 'unique'
        elif col.name.endswith('id') or unique_count <= 20:
            column_type = 'category'
        elif numeric:
            column_type = 'number'
        else:
            column_type = 'unknown'