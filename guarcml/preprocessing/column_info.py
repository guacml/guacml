ColType = Enum('ColType', 'ID CATEGORICAL NUMERIC')
Cardinality = Enum('Cardinality', 'LOW MEDIUM HIGH')

class ColumnInfo:

    def __init__(self, type, cardinality):
        self.type = type
        self.cardinality = cardinality


