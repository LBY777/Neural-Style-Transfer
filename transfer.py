import sys
assert len(sys.argv) == 3

import tensorflow as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import keras.backend as K
from keras.utils import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()
from datetime import datetime
from PIL import Image


def to_avgpool(shape, nconvs=None):
    if nconvs != None:
        assert(1<=nconvs<=13)
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    i = vgg.input
    x = i
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            x = AveragePooling2D()(x)
        else:
            x = layer(x)
    mod = Model(i, x)
    if nconvs == None:
        return mod
    output, n = None, 0
    for layer in mod.layers:
        if layer.__class__ == Conv2D:
            n += 1
        if n >= nconvs:
            output = layer.output
            break
    return Model(mod.input, output)
def img_scale(x):
    x -= x.min()
    return x / x.max()
def gram_matrix(img):
  X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
  G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
  return G
def loss_style(y, t):
  return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))
def img_unpreprocess(img):
  img[..., 0] += 103.939
  img[..., 1] += 116.779
  img[..., 2] += 126.68
  return img[..., ::-1]

def minimize(fn, epochs, batch_shape):
  t0 = datetime.now()
  losses = []
  x = np.random.randn(np.prod(batch_shape))
  for i in range(epochs):
    x, l, _ = fmin_l_bfgs_b(
      func=fn,
      x0=x,
      maxfun=20
    )
    x = np.clip(x, -127, 127)
    print("iter=%s, loss=%s" % (i, l))
    losses.append(l)

  print("duration:", datetime.now() - t0)
  plt.plot(losses)
  plt.show()

  newimg = x.reshape(*batch_shape)
  final_img = img_unpreprocess(newimg)
  return final_img[0]
def load_preprocess(path, shape=None):
  img = load_img(path, target_size=shape)
  x = img_to_array(img)
  x = np.expand_dims(x, axis=0)
  return preprocess_input(x)


content = sys.argv[1]
style = sys.argv[2]
def combine_style(content, style, epochs=3):
  content_img = load_preprocess(content)
  h, w = content_img.shape[1:3]
  style_img = load_preprocess(style, (h, w))
  batch_shape = content_img.shape
  shape = content_img.shape[1:]
  vgg = to_avgpool(shape)
  content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
  content_target = K.variable(content_model.predict(content_img))
  symbolic_conv_outputs = [
    layer.get_output_at(1) for layer in vgg.layers \
    if layer.name.endswith('conv1')
  ]
  style_model = Model(vgg.input, symbolic_conv_outputs)
  style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]
  style_weights = [0.2,0.4,0.3,0.5,0.2]
  loss = K.mean(K.square(content_model.output - content_target))
  for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
    loss += w * loss_style(symbolic[0], actual[0])
  grads = K.gradients(loss, vgg.input)
  loss_and_grads = K.function(
    inputs=[vgg.input],
    outputs=[loss] + grads
  )
  print()
  print("*"*20, "Started painting", "*"*20)
  print("Style Image: ", style)
  print("Content Image: ", content)
  print()
  def loss_and_grads_wrapper(x_vec):
    l, g = loss_and_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)
  return minimize(loss_and_grads_wrapper, epochs, batch_shape)
out = combine_style(content, style)

saved_img = Image.fromarray(np.uint8(out))
name = '{content[:-4]}+{style[:-4]}.jpg'
saved_img.save(f'./painted/{content[:-4]}+{style[:-4]}.jpg')
print("*"*20, "Painting Completed", "*"*20)
print(f"Painted image stored as {name} in folder painted")
