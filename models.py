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
    由于cross_network模块没有激活函数, 因此为避免输出出现无穷大的情况, 输入数据应该先进行scaler
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
    pre_out = layers.Concatenate(axis=-1)([cross_out, deep_out])
    # out = deep_out
    pre_out = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg)(pre_out)
    out = tf.nn.sigmoid(pre_out)
    # 生成并返回模型
    model = keras.Model(inputs, out)
    return model

def create_xDeepFM(df, settings):
    """
    在 WideDeep 的基础上增加了 cin 模块
    """
    reg = settings.reg
    activation = settings.act
    cin_size = settings.cin_size

    inputs, (dense_inputs, cat_weight, cat_embed) = process( df, settings)
    # 逻辑回归层
    logistic_layer = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, name='logistic_layer')
    linear_out = logistic_layer(dense_inputs)
    linear_out = layers.Add()([linear_out, cat_weight])
    # cin层
    cin_layer = CIN_Layer(cin_size=cin_size, reg=reg)
    cin_out = cin_layer(cat_embed)
    # 数值特征拼接 类别特征的 embedding 作为 deep_layer 的输入
    deep_layer = keras.Sequential([
        layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in settings.units
    ])
    embed_input = layers.Flatten()(cat_embed)
    deep_input = layers.Concatenate(axis=-1)([dense_inputs, embed_input])
    deep_out = deep_layer(deep_input)
    # 拼接之后在经过一个逻辑回归层得到输出
    pre_out = layers.Concatenate(axis=-1)([linear_out, cin_out, deep_out])
    out = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg)(pre_out)
    out = keras.activations.sigmoid(out)
    # 生成并返回模型
    model = keras.Model(inputs, out)
    return model

def create_FNN(df, settings, activation='relu'):

    reg = settings.reg
    activation = settings.act

    inputs, (dense_inputs, cat_weight, cat_embed) = process( df, settings)
    # 逻辑回归层
    logistic_layer = layers.Dense(1, kernel_regularizer=settings.reg, bias_regularizer=settings.reg)
    linear_out = logistic_layer(dense_inputs)
    # fm层
    fm_layer = FMLayer(name='FMLayer')
    fm_out = fm_layer(cat_embed)
    # 逻辑回归层拆分成了 数值特征的逻辑回归层 以及 类别特征的 weight 层， 在此将他们加起来
    fm_out = layers.Add()([linear_out, cat_weight, fm_out])
    fm_out = keras.activations.sigmoid(fm_out)
    # 生成并返回 FM 模型
    FM_Model = keras.Model(inputs, fm_out)

    embed_input = layers.Flatten()(cat_embed)
    deep_input = layers.Concatenate(axis=-1)([dense_inputs, embed_input])
    # deep_network
    deep_layer = keras.Sequential([
        layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in settings.units
    ])
    deep_out = deep_layer(deep_input)

    out = layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, activation='sigmoid')(deep_out)
    # fnn model
    FNN_Model = keras.Model(inputs, out)
    return FM_Model, FNN_Model

def create_PNN(df, settings):

    reg = settings.reg
    activation = settings.act
    units = settings.units
    MODE = settings.PNN_MODE

    inputs, (dense_inputs, cat_weight, cat_embed) = process( df, settings)

    embed_input = keras.layers.Flatten()(cat_embed)

    if MODE == 'inner':
        pro_out = InnerProductLayer()(cat_embed)
        deep_input = layers.Concatenate(axis=-1)([dense_inputs, embed_input, pro_out])
    elif MODE == 'out':
        pro_out = OutProductLayer(reg)(cat_embed)
        deep_input = layers.Concatenate(axis=-1)([dense_inputs, embed_input, pro_out])
    elif MODE == 'both':
        inner_out = InnerProductLayer()(cat_embed)
        out_out = OutProductLayer(reg)(cat_embed)
        deep_input = layers.Concatenate(axis=-1)([dense_inputs, embed_input, inner_out, out_out])

    deep_layer = keras.Sequential([
        keras.layers.Dense(i, kernel_regularizer=reg, bias_regularizer=reg, activation=activation) for i in units
    ])
    deep_layer.add( keras.layers.Dense(1, kernel_regularizer=reg, bias_regularizer=reg, activation='sigmoid'))

    out = deep_layer(deep_input)
    model = keras.Model(inputs, out)
    return model