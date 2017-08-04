import marshal
import importlib


def deep_update(left, right):
    for key, value in right.items():
        if isinstance(value, dict) and key in left:
            deep_update(left[key], value)
        else:
            left[key] = value


def deep_copy(obj):
    return marshal.loads(marshal.dumps(obj))


def get_fully_qualified_class_name(obj):
    return obj.__module__ + '.' + obj.__class__.__name__


def get_class_from_string(string):
    module_path, class_name = string.rsplit('.', 1)
    module = importlib.import_module(module_path)

    return getattr(module, class_name)
