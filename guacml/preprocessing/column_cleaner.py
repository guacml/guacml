if n_blanks > 0:
    not_null = not_null.replace('', None)
    print('Column {0}: Replaced {1} blank values with None.'.format(col_name, n_blanks))