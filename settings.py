from tensorflow import keras

class Settings:
    # Feature names
    dense_feature = ['I' + str(i) for i in range(1,14)]
    cat_feature = ['C'+ str(i) for i in range(1,27)]

    # embedding
    embed_mode = 'vector'
    embed_size = 8

    # optimizer
    opt = keras.optimizers.Adam()

    # Regularizer
    REG = 1e-4
    reg = keras.regularizers.L2(REG)