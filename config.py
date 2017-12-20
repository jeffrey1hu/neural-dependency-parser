class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_features = 36
    n_classes = 3
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 20
    lr = 0.001
    reg = 1e-8
