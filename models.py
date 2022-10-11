import tensorflow as tf
from tensorflow import keras
from settings import Settings

from process import process
from layers import FMLayer, FFMLayer, Cross_Layer, CIN_Layer,InnerProductLayer, OutProductLayer

def create_FM(df, settings):

    inputs, (dense_inputs, cat_weight, cat_embed) = process(df,settings)

    logistic_layer = keras.layers.Dense(1, kernel_regularizer=settings.reg, bias_regularizer=settings.reg, name='logistic_layer')
    fm_layer = FMLayer(name='FMLayer')

    linear_out = logistic_layer(dense_inputs)
    fm_out = fm_layer(cat_embed)

    out = tf.reduce_sum([linear_out, cat_weight, fm_out], axis=0)
    out = tf.nn.sigmoid(out)

    model = keras.Model(inputs, out)
    return model

def create_FFM(df, num_feature_names=None,cat_feature_names=None,EMBED_SIZE=10,reg=1e-4):
    
    reg = keras.regularizers.L2(l2=reg)

    inputs, (num_input, num_embed),(cat_onehot, cat_embed)= preprocess(df,
                                                     num_feature_names=num_feature_names,
                                                     cat_feature_names=cat_feature_names,
                                                     embed_size=EMBED_SIZE,
                                                     embed_mode='matrix',
                                                     reg=reg
                                                    )
    linear_input = tf.concat([num_input, cat_onehot], axis=-1)
    embed_input = tf.concat([num_embed, cat_embed], axis=1)
    wide_layer = keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, name='wide_layer')
    ffm_layer = FFMLayer(name='FFMLayer')

    linear_out = wide_layer(linear_input)
    ffm_out = ffm_layer(embed_input)

    out = tf.reduce_sum([linear_out, ffm_out], axis=0)
    out = tf.nn.sigmoid(out)

    model = keras.Model(inputs, out)
    return model

def create_WideDeep(df, num_feature_names=None, cat_feature_names=None, units=[256,128], EMBED_SIZE=10,reg=1e-4, activation='relu'):

    reg = keras.regularizers.L2(l2=reg)

    inputs, (num_input, num_embed),(cat_onehot, cat_embed) = preprocess(df,
                                                 num_feature_names=num_feature_names,
                                                 cat_feature_names=cat_feature_names,
                                                 embed_size=EMBED_SIZE,
                                                 embed_mode='vector',
                                                 reg=reg
                                                )
    
    linear_input = tf.concat([num_input, cat_onehot], axis=-1)
    embed_input = tf.concat([num_embed, cat_embed], axis=1)

    embed_input = keras.layers.Flatten()(embed_input)

    wide_layer = keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, name='wide_layer')
    deep_layer = keras.Sequential([
        keras.layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in units
    ])
    deep_layer.add( keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg))

    wide_out = wide_layer(linear_input)
    deep_out = deep_layer(embed_input)

    out = tf.reduce_sum([wide_out, deep_out], axis=0)
    out = tf.nn.sigmoid(out)

    model = keras.Model(inputs, out)
    return model

def create_DeepFM(df, num_feature_names=None, cat_feature_names=None, units=[256,128], EMBED_SIZE=10,reg=1e-4, activation='relu'):

    reg = keras.regularizers.L2(l2=reg)

    inputs, (num_input, num_embed),(cat_onehot, cat_embed) = preprocess(df,
                                                 num_feature_names=num_feature_names,
                                                 cat_feature_names=cat_feature_names,
                                                 embed_size=EMBED_SIZE,
                                                 embed_mode='vector',
                                                 reg=reg
                                                )
    
    linear_input = tf.concat([num_input, cat_onehot], axis=-1)
    embed_input = tf.concat([num_embed, cat_embed], axis=1)

    linear_layer = keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, name='wide_layer')
    fm_layer = FMLayer(name='FMLayer')

    linear_out = linear_layer(linear_input)
    fm_out = fm_layer(embed_input)

    embed_input_flatten = keras.layers.Flatten()(embed_input)
    deep_layer = keras.Sequential([
        keras.layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in units
    ])
    deep_layer.add( keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg))
    deep_out = deep_layer(embed_input_flatten)

    out = tf.reduce_sum([linear_out, fm_out, deep_out], axis=0)
    out = tf.nn.sigmoid(out)

    model = keras.Model(inputs, out)
    return model

def create_DeepCross(df, num_feature_names=None, cat_feature_names=None, layer_num=5, units = [256, 128, 64], EMBED_SIZE=10,reg=1e-4, activation='relu'):

    reg = keras.regularizers.L2(l2=reg)

    inputs, (num_input, _),(_, cat_embed) = preprocess(df,
                                                 num_feature_names=num_feature_names,
                                                 cat_feature_names=cat_feature_names,
                                                 embed_size=EMBED_SIZE,
                                                 embed_mode='vector',
                                                 reg=reg
                                                )
    
    cat_embed = keras.layers.Flatten()(cat_embed)
    num_catEmbed = tf.concat([num_input, cat_embed], axis=-1)

    cross_layer = Cross_Layer(layer_num, reg)
    cross_out = cross_layer(num_catEmbed)

    deep_layer = keras.Sequential([
        keras.layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in units
    ])
    deep_layer.add( keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg))
    deep_out = deep_layer(num_catEmbed)

    out = tf.concat([cross_out, deep_out], axis=-1)
    out = keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, activation=None)(out)
    out = tf.nn.sigmoid(out)

    model = keras.Model(inputs, out)
    return model

def create_xDeepFM(df, num_feature_names=None, cat_feature_names=None, cin_size=[64,64], units = [256, 128, 64], EMBED_SIZE=10,reg=1e-4, activation='relu'):

    reg = keras.regularizers.L2(l2=reg)

    inputs, (num_input, num_embed),(cat_onehot, cat_embed) = preprocess(df,
                                                 num_feature_names=num_feature_names,
                                                 cat_feature_names=cat_feature_names,
                                                 embed_size=EMBED_SIZE,
                                                 embed_mode='vector',
                                                 reg=reg
                                                )
    
    linear_input = tf.concat([num_input, cat_onehot], axis=-1)

    cin_layer = CIN_Layer(cin_size, reg)
    cin_out = cin_layer(cat_embed)

    deep_layer = keras.Sequential([
        keras.layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in units
    ])

    cat_embed = keras.layers.Flatten()(cat_embed)
    num_catEmbed = tf.concat([num_input, cat_embed], axis=-1)
    deep_out = deep_layer(num_catEmbed)

    linear_layer = keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, name='wide_layer')
    linear_out = linear_layer(linear_input)

    out = tf.concat([cin_out, deep_out, linear_out], axis=-1)
    out = keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, activation='sigmoid')(out)

    model = keras.Model(inputs, out)
    return model

def create_FNN(df, num_feature_names=None, cat_feature_names=None, units = [256, 128, 64], EMBED_SIZE=10,reg=1e-4, activation='relu'):

    reg = keras.regularizers.L2(l2=reg)

    inputs, (num_input, num_embed),(cat_onehot, cat_embed) = preprocess(df,
                                                 num_feature_names=num_feature_names,
                                                 cat_feature_names=cat_feature_names,
                                                 embed_size=EMBED_SIZE,
                                                 embed_mode='vector',
                                                 reg=reg
                                                )
    
    linear_input = tf.concat([num_input, cat_onehot], axis=-1)
    embed_input = tf.concat([num_embed, cat_embed], axis=1)

    #FM
    wide_layer = keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, name='wide_layer')
    fm_layer = FMLayer(name='FMLayer')

    linear_out = wide_layer(linear_input)
    fm_out = fm_layer(embed_input)

    FM_Model_out = tf.reduce_sum([linear_out, fm_out], axis=0)
    FM_Model_out = tf.nn.sigmoid(FM_Model_out)

    #DNN
    embed_input_nn = keras.layers.Flatten()(embed_input)
    deep_layer = keras.Sequential([
        keras.layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in units
    ])
    deep_layer.add( keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg))

    deep_out = deep_layer(embed_input_nn)
    deep_out = tf.nn.sigmoid(deep_out)

    # fm model
    FM_Model = keras.Model(inputs, FM_Model_out)
    # fnn model
    FNN_Model = keras.Model(inputs, deep_out)
    return FM_Model, FNN_Model

def create_PNN(df, num_feature_names=None, cat_feature_names=None, MODE='inner',units=[256,128], EMBED_SIZE=10,reg=1e-4, activation='relu'):

    reg = keras.regularizers.L2(l2=reg)

    inputs, (num_input, num_embed),(cat_onehot, cat_embed) = preprocess(df,
                                                 num_feature_names=num_feature_names,
                                                 cat_feature_names=cat_feature_names,
                                                 embed_size=EMBED_SIZE,
                                                 embed_mode='vector',
                                                 reg=reg
                                                )
    
    embed_input = tf.concat([num_embed, cat_embed], axis=1)

    if MODE == 'inner':
        pro_layer = InnerProductLayer()
    elif MODE == 'out':
        pro_layer = OutProductLayer(reg)
    
    pro_out = pro_layer(embed_input)

    embed_flatten = keras.layers.Flatten()(embed_input)

    deep_layer = keras.Sequential([
        keras.layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in units
    ])
    deep_layer.add( keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg))

    deep_input = tf.concat([embed_flatten,pro_out], axis=1)

    deep_out = deep_layer(deep_input)

    out = tf.nn.sigmoid(deep_out)

    model = keras.Model(inputs, out)
    return model