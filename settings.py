from tensorflow import keras

class Settings:
    # Feature names
    dense_feature = ['I' + str(i) for i in range(1,14)]
    cat_feature = ['C'+ str(i) for i in range(1,27)]

    # embedding
    embed_mode = 'vector'
    embed_size = 8

    # deep_layers
    units = [256,128,64]
    # cross_layer_numb
    layer_num = 6
    # cin_size
    cin_size = [ 32, 32, 32]
    # PNN_MODE
    PNN_MODE = 'both'

    # deep layers activation
    act = 'relu'

    # optimizer
    opt = keras.optimizers.Adam()

    # Regularizer
    REG = 1e-4
    reg = keras.regularizers.L2(REG)