def deep_update(left, right):
    for key, value in right.items():
        if isinstance(value, dict) and key in left:
            deep_update(left[key], value)
        else:
            left[key] = value
