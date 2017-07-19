from guacml.target_transforms.log_transform import LogTransform


def target_transform_from_name(transform_name):
    if transform_name == 'log':
        return LogTransform()
    else:
        raise ValueError('Unknown target transformation name ' + transform_name)
