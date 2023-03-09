from model.LCNet import LCNet


def build_model(model_name, num_classes):
    if model_name == 'LCNet':
        return LCNet(classes=num_classes)
