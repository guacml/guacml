from enum import Enum

ProblemType = Enum('ProblemType', 'BINARY_CLAS MULTI_CLAS REGRESSION')

ColType = Enum('ColType', 'BINARY NUMERIC ORDINAL INT_ENCODING\
                           ID CATEGORICAL DATETIME TEXT WORDS UNKNOWN')


