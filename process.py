import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def stack_dict(inputs, fun=tf.stack):
    # cast dtype and stack tensor data
    value = []
    for key in inputs.keys():
        value.append(tf.cast(inputs[key], tf.float32))

    return fun(value,axis=-1)

def process(df, settings):
    #create input layers
    inputs = {}
    dense_inputs = {}
    for name, column in df.items():
        if type(column[0]) == str:
            dtype = tf.string
        elif type(column[0]) == int:
            dtype = tf.int64
        else:
            dtype = tf.float32
        inputs[name] = keras.Input(shape=(), name=name, dtype=dtype)
        if name in settings.dense_feature:
            dense_inputs[name] = inputs[name]

    dense_inputs = stack_dict(dense_inputs)
    
    # # Create OneHot and Embedding for Category Feature
    cat_weight, cat_embed = createCatEmbed(
        df=df,
        inputs=inputs,
        settings=settings
    )
    
    return inputs, (dense_inputs, cat_weight, cat_embed)

def createCatEmbed(df, inputs, settings):
    # Create OneHot and Embedding for Category Feature
    cat_weight = []
    cat_embed = []
    feature_size = len(settings.cat_feature)
    for name in settings.cat_feature:
        vocabulary = list(set(df[name]))
        cat_size = len(vocabulary)
        
        lookup_int = keras.layers.StringLookup(vocabulary=vocabulary, output_mode='int', name=name+'_int')
        
        weight = keras.layers.Embedding(
            input_dim = cat_size + 1,
            output_dim = 1,
            embeddings_regularizer = settings.reg,
            name = name + '_weight'
        )
        if settings.embed_mode == 'vector':
            embed = keras.layers.Embedding(input_dim=cat_size+1, 
                                           output_dim=settings.embed_size,
                                           embeddings_regularizer=settings.reg,
                                           name=name+'_embed',
                                          )
        elif settings.embed_mode == 'matrix':
            embed = CatEmbed2D(shape=(cat_size+1,feature_size,settings.embed_size),
                                    reg=settings.reg,
                                    name=name+'_matrixEmbed',)
        
        x = lookup_int(inputs[name])
        x_1 = weight(x)
        x_2 = embed(x)
        cat_weight.append(x_1)
        cat_embed.append(x_2)
    cat_weight = keras.layers.Add()(cat_weight)
    cat_embed = tf.stack(cat_embed, axis=1)
    
    return cat_weight, cat_embed

class CatEmbed2D(keras.layers.Layer):
    # category matrix embed layer, return enmbed
    def __init__(self, shape, reg, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.reg = reg

    def build(self, input_shape):
        self.embed_weight = self.add_weight(
            shape=self.shape,
            initializer='random_normal',
            regularizer=self.reg,
            trainable=True
        )

    def call(self, inputs):
        return tf.gather(self.embed_weight, inputs)