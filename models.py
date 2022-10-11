import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from process import process
from layers import FMLayer, FFMLayer, Cross_Layer, CIN_Layer,InnerProductLayer, OutProductLayer

def create_FM(df, settings):
    """
    生成FM模型
    """
    # 生成输入层及后续的 dense_inputs, cat_weight, cat_embed
    inputs, (dense_inputs, cat_weight, cat_embed) = process(df,settings)
    # 逻辑回归层
    logistic_layer = layers.Dense(1, kernel_regularizer=settings.reg, bias_regularizer=settings.reg, name='logistic_layer')
    linear_out = logistic_layer(dense_inputs)
    # fm层
    fm_layer = FMLayer(name='FMLayer')
    fm_out = fm_layer(cat_embed)
    # 逻辑回归层拆分成了 数值特征的逻辑回归层 以及 类别特征的 weight 层， 在此将他们加起来
    out = layers.Add()([linear_out, cat_weight, fm_out])
    out = keras.activations.sigmoid(out)
    # 生成并返回模型
    model = keras.Model(inputs, out)
    return model

def create_FFM(df, settings):
    """
    与FM模型唯一的不同在于把FM层换成了FFM层
    """
    # FFM 模型的embed类型为矩阵
    settings.embed_mode = 'matrix'
    # 生成输入层及后续的 dense_inputs, cat_weight, cat_embed
    inputs, (dense_inputs, cat_weight, cat_embed) = process(df,settings)
    # 逻辑回归层
    logistic_layer = layers.Dense(1, kernel_regularizer=settings.reg, bias_regularizer=settings.reg, name='logistic_layer')
    linear_out = logistic_layer(dense_inputs)
    # ffm层
    ffm_layer = FFMLayer(name='FMLayer') 
    ffm_out = ffm_layer(cat_embed)
    # 逻辑回归层拆分成了 数值特征的逻辑回归层 以及 类别特征的 weight 层， 在此将他们加起来
    out = layers.Add()([linear_out, cat_weight, ffm_out])
    out = keras.activations.sigmoid(out)
    # 生成并返回模型
    model = keras.Model(inputs, out)
    return model

def create_WideDeep(df, settings):
    """
    与FM的不同之处在于用 deep_layer 替换了 fm_layer
    """
    reg = settings.reg
    activation = settings.act

    inputs, (dense_inputs, cat_weight, cat_embed) = process( df, settings)
    # 逻辑回归层
    logistic_layer = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, name='logistic_layer')
    linear_out = logistic_layer(dense_inputs)
    # 数值特征拼接 类别特征的 embedding 作为 deep_layer 的输入
    deep_layer = keras.Sequential([
        layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in settings.units
    ])
    embed_input = layers.Flatten()(cat_embed)
    deep_input = layers.Concatenate(axis=-1)([dense_inputs, embed_input])
    deep_out = deep_layer(deep_input)
    deep_out = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg)(deep_out)
    # 逻辑回归层拆分成了 数值特征的逻辑回归层 以及 类别特征的 weight 层， 在此将他们加起来
    out = layers.Add()([linear_out, cat_weight, deep_out])
    out = keras.activations.sigmoid(out)
    # 生成并返回模型
    model = keras.Model(inputs, out)
    return model

def create_DeepFM(df, settings):
    """
    相较于FM模型, 增加了deep 层
    相较于WideDeep模型, 增加了 FM 层
    """
    reg = settings.reg
    activation = settings.act

    inputs, (dense_inputs, cat_weight, cat_embed) = process( df, settings)
    # 逻辑回归层
    logistic_layer = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, name='logistic_layer')
    linear_out = logistic_layer(dense_inputs)
    # fm层
    fm_layer = FMLayer(name='FMLayer')
    fm_out = fm_layer(cat_embed)
    # 数值特征拼接 类别特征的 embedding 作为 deep_layer 的输入
    deep_layer = keras.Sequential([
        layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in settings.units
    ])
    embed_input = layers.Flatten()(cat_embed)
    deep_input = layers.Concatenate(axis=-1)([dense_inputs, embed_input])
    deep_out = deep_layer(deep_input)
    deep_out = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg)(deep_out)
    # 逻辑回归层(数值特征、类别特征)、fm层、deep层
    out = layers.Add()([linear_out, cat_weight, fm_out, deep_out])
    out = keras.activations.sigmoid(out)
    # 生成并返回模型
    model = keras.Model(inputs, out)
    return model

def create_DeepCross(df, settings):
    """
    两个模块分别为 deep_network 和 cross_network
    两个模块共享输入
    """
    reg = settings.reg
    activation = settings.act
    layer_num = settings.layer_num
    # 只需要对类别特征进行 embedding 操作即可， 随后与数值特征结合生成两个模块的输入
    inputs, (dense_inputs, _, cat_embed) = process( df, settings)
    embed_input = layers.Flatten()(cat_embed)
    deep_input = layers.Concatenate(axis=-1)([dense_inputs, embed_input])    
    # Cross_Layer层
    cross_layer = Cross_Layer(layer_num, reg)
    cross_out = cross_layer(deep_input)
    # deep_network
    deep_layer = keras.Sequential([
        layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in settings.units
    ])
    deep_out = deep_layer(deep_input)
    # 逻辑回归层(数值特征、类别特征)、fm层、deep层
    pre_out = layers.Concatenate(axis=-1)([cross_out, deep_out])
    # out = deep_out
    pre_out = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg)(pre_out)
    out = tf.nn.sigmoid(pre_out)
    out = tf.clip_by_value( out, 1e-10, 0.99999)
    # 生成并返回模型
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