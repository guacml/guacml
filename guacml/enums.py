from enum import Enum

# pylint: disable=invalid-name
ProblemType = Enum('ProblemType', 'BINARY_CLAS MULTI_CLAS REGRESSION')
ColType = Enum('ColType', 'BINARY NUMERIC ORDINAL INT_ENCODING\
                           ID CATEGORICAL DATETIME TEXT WORDS LIST UNKNOWN')
