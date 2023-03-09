from model.LMFFNet import LMFFNet


def build_model(model_name, num_classes):
    if model_name == 'LMFFNet':
        return LMFFNet(classes=num_classes)
