import keras
from keras.models import Model
from keras import layers
import keras.backend as K
import tensorflow as tf
import numpy as np

class DisparityRegression(layers.Layer):
  def call(self, x):
    pos = np.arange(int(x.shape[1]), dtype=np.float32)
    arr = tf.zeros(tf.shape(x))+(np.zeros((x.shape[1],x.shape[2],x.shape[3],x.shape[4]))+pos[:,np.newaxis,np.newaxis,np.newaxis])
    x *= arr
    return x

class ShiftRight(layers.Layer):
  def __init__(self, shiftcount, isLeft, **kwargs):
    self.shiftcount = shiftcount
    self.isLeft = isLeft
    super(ShiftRight, self).__init__(**kwargs)

  def call(self, right):
    if (self.isLeft == True): ##Remove first few rows of left
      x = tf.concat([tf.zeros([tf.shape(right)[0],right.shape[1],self.shiftcount,right.shape[3]]),right[:,:,self.shiftcount:]],axis=2)
    else: ##Remove last few rows of right
      x = tf.concat([tf.zeros([tf.shape(right)[0],right.shape[1],self.shiftcount,right.shape[3]]),right[:,:,:-self.shiftcount]],axis=2)
    return x

def conv2(x,filter_count,kernel_size=(3,3),stride=1,dilation=(1,1),padding='same',use_bias=True,alpha=0.2,ifrelu=True,ifNorm=True,ifUpsample=False,output_padding=[0,0]):
  if (ifUpsample == True):
    x = layers.Conv2DTranspose(filters=filter_count,kernel_size=kernel_size,strides=(stride,stride),padding=padding,use_bias=use_bias,output_padding=output_padding)(x) 
  else:
    x = layers.Conv2D(filters=filter_count,kernel_size=kernel_size,strides=(stride,stride),dilation_rate=dilation,padding=padding,use_bias=use_bias)(x)
  if (ifNorm == True):
    x = layers.BatchNormalization()(x)
  if (ifrelu == True):
    layers.LeakyReLU(alpha=alpha)(x)
  return x

def conv3(x,filter_count,kernel_size=(3,3,3),stride=1,padding='same',use_bias=True,alpha=0.2,ifrelu=True,ifNorm=True,ifUpsample=False,output_padding=[0,0,0]):
  if (ifUpsample == True):
    x = layers.UpSampling3D(size=(stride,stride,stride))(x)
    x = layers.Conv3D(filters=filter_count,kernel_size=kernel_size,strides=(1,1,1),padding=padding,use_bias=use_bias)(x)
  else:
    x = layers.Conv3D(filters=filter_count,kernel_size=kernel_size,strides=(stride,stride,stride),padding=padding,use_bias=use_bias)(x)
  if (ifNorm == True):
    x = layers.BatchNormalization()(x)
  if (ifrelu == True):
    layers.LeakyReLU(alpha=alpha)(x)
  return x

def afterConv(x):
  x = layers.BatchNormalization()(x)
  x = layers.LeakyReLU(alpha=0.2)(x)
  return x

def makeBranch(x,branchx,pool_size,H=128,W=512):
  branch = layers.AveragePooling2D(pool_size=(pool_size,pool_size))(x)
  branch = branchx(branch)
  branch = afterConv(branch)
  branch = layers.Lambda( 
            lambda image: tf.image.resize( 
              image, (H//4, W//4), 
              method = tf.image.ResizeMethod.BILINEAR,
              preserve_aspect_ratio = False
            )
          )(branch)
  return branch

def SPP(x,y,branch1,branch2,branch3,branch4,fusion1,fusion2,max_pool_size=32,H=128,W=512):
  branch1 = makeBranch(x,branch1,max_pool_size,H=H,W=W)
  branch2 = makeBranch(x,branch2,int(max_pool_size/2),H=H,W=W)
  branch3 = makeBranch(x,branch3,int(max_pool_size/4),H=H,W=W)
  branch4 = makeBranch(x,branch4,int(max_pool_size/8),H=H,W=W)
  
  fusion = layers.Concatenate()([x,y,branch1,branch2,branch3,branch4])
  fusion = fusion1(fusion)
  fusion = afterConv(fusion)
  fusion = fusion2(fusion)
  fusion = afterConv(fusion)
  return fusion

def cost_volume(x,y):
  return layers.Concatenate()([x,y])

def ResLayer(x,filter_count):
  y = conv3(x,filter_count)
  y = conv3(y,filter_count)
  y = layers.add([y,x])
  return y

def CNN3D_Basic(x,filter_count,upSampleStride=4):
  x = conv3(x,filter_count)
  x = conv3(x,filter_count)
  for i in range(4):
    x = ResLayer(x,filter_count)

  x = conv3(x,filter_count)
  x = conv3(x,1)
  x = conv3(x,filter_count=1,kernel_size=(5,5,5),stride=upSampleStride,padding='same',ifrelu=False,ifNorm=False,ifUpsample=True)
  x = layers.Softmax(axis=1)(x)
  x = DisparityRegression()(x)
  x = layers.Lambda(lambda xin: K.sum(xin, axis=1))(x)
  return x

def CNN3D(x,filter_count,upSampleStride=4):
  x = conv3(x,filter_count)
  x = conv3(x,filter_count)
  y = conv3(x,filter_count)
  y = conv3(y,filter_count)

  x = layers.add([x,y])
  y = conv3(x,filter_count,stride=2) ## 1/8
  y = conv3(y,filter_count)

  z = conv3(y,filter_count,stride=2) ## 1/16
  z = conv3(z,filter_count)
  z = conv3(z,filter_count,stride=2,ifUpsample=True) ## 1/8
  z = layers.add([z,y])

  a = conv3(z,filter_count,stride=2,ifUpsample=True) ##1/4
  a = layers.add([a,x])

  aside_1 = conv3(a,filter_count)
  aside_1 = conv3(aside_1,1)
  output1 = conv3(aside_1,filter_count=1,kernel_size=(5,5,5),stride=upSampleStride,padding='same',ifrelu=False,ifNorm=False,ifUpsample=True)
  output1 = layers.Softmax(axis=1)(output1)
  output1 = DisparityRegression()(output1)
  output1 = layers.Lambda(lambda xin: K.sum(xin, axis=1))(output1)

  a = conv3(a,filter_count,stride=2) ## 1/8
  a = conv3(a,filter_count)
  z = layers.add([a,z])

  z = conv3(z,filter_count,stride=2) ## 1/16
  z = conv3(z,filter_count)
  z = conv3(z,filter_count,stride=2,ifUpsample=True) ## 1/8
  z = layers.add([z,y])

  a = conv3(z,filter_count,stride=2,ifUpsample=True) ## 1/4
  a = layers.add([a,x])

  aside_2 = conv3(a,filter_count)
  aside_2 = conv3(aside_2,1)
  aside_2 = layers.add([aside_2,aside_1])
  output2 = conv3(aside_2,filter_count=1,kernel_size=(5,5,5),stride=upSampleStride,padding='same',ifrelu=False,ifNorm=False,ifUpsample=True)
  output2 = layers.Softmax(axis=1)(output2)
  output2 = DisparityRegression()(output2)
  output2 = layers.Lambda(lambda xin: K.sum(xin, axis=1))(output2)

  a = conv3(a,filter_count,stride=2) ## 1/8
  a = conv3(a,filter_count)
  z = layers.add([a,z])
  z = conv3(z,filter_count,stride=2) ## 1/16
  z = conv3(z,filter_count)
  z = conv3(z,filter_count,stride=2,ifUpsample=True) ## 1/8
  y = layers.add([z,y])

  y = conv3(y,filter_count,stride=2,ifUpsample=True) ## 1/4
  x = layers.add([y,x])

  aside_3 = conv3(x,filter_count)
  aside_3 = conv3(aside_3,1)
  aside_3 = layers.add([aside_3,aside_2])
  output3 = conv3(aside_3,filter_count=1,kernel_size=(5,5,5),stride=upSampleStride,padding='same',ifrelu=False,ifNorm=False,ifUpsample=True)
  output3 = layers.Softmax(axis=1)(output3)
  output3 = DisparityRegression()(output3)
  output3 = layers.Lambda(lambda xin: K.sum(xin, axis=1))(output3)
  return [output1,output2,output3]

def sharedCNN(x, layers = [], arraysOfLayers = []):
  conv2_16 = x
  for i in layers:
    x = i(x)
    x = afterConv(x)
  for i in arraysOfLayers:
    for j in range(len(i)):
      x = i[j](x)
      x = afterConv(x)
      if j == 15:
        conv2_16 = x
  conv4_3 = x
  return [conv2_16,conv4_3]

def PSMN(left_input,right_input,disparity = -1,shiftcount=4, base_filter_count = 32, kernel_size = 3, use_bias = True, basic3DCNN = False):
  if left_input.shape != right_input.shape:
    print("Error: Left and Right Input shape must be the same")
    break
  if disparity==-1:
    disparity = right_input.shape[2]//4 - (right_input.shape[2]//4%4)
  if disparity%4 != 0:
    print("Disparity must be multiple of four")
    break
  
  ##CNN Layers
  conv0_1 = layers.Conv2D(filters=base_filter_count,kernel_size=kernel_size,strides=(2,2),dilation_rate=(1,1),padding='same',use_bias=use_bias)
  conv0_2 = layers.Conv2D(filters=base_filter_count,kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias)
  conv0_3 = layers.Conv2D(filters=base_filter_count,kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias)
  conv1x = []
  conv2x = []
  conv3x = []
  conv4x = []

  for i in range(3):
    conv1x.append(layers.Conv2D(filters=base_filter_count,kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias))

  conv2x.append(layers.Conv2D(filters=int(base_filter_count*2),kernel_size=kernel_size,strides=(2,2),dilation_rate=(1,1),padding='same',use_bias=use_bias))
  for i in range(1,16):
    conv2x.append(layers.Conv2D(filters=int(base_filter_count*2),kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias))

  for i in range(3):
    conv3x.append(layers.Conv2D(filters=int(base_filter_count*4),kernel_size=kernel_size,strides=(1,1),dilation_rate=(2,2),padding='same',use_bias=use_bias))
  
  for i in range(3):
    conv4x.append(layers.Conv2D(filters=int(base_filter_count*4),kernel_size=kernel_size,strides=(1,1),dilation_rate=(4,4),padding='same',use_bias=use_bias))

  ###SPP LAYERS
  branch_1 = layers.Conv2D(filters=base_filter_count,kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias)
  branch_2 = layers.Conv2D(filters=base_filter_count,kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias)
  branch_3 = layers.Conv2D(filters=base_filter_count,kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias)
  branch_4 = layers.Conv2D(filters=base_filter_count,kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias)
  fusion1 = layers.Conv2D(filters=int(base_filter_count*4),kernel_size=kernel_size,strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias)
  fusion2 = layers.Conv2D(filters=base_filter_count,kernel_size=(1,1),strides=(1,1),dilation_rate=(1,1),padding='same',use_bias=use_bias)

  ##Make Cost Volume
  costvolume_arr = []
  left = left_input
  right = right_input
  for i in range(disparity//shiftcount): 
    if i != 0:
      right = ShiftRight(shiftcount=shiftcount,isLeft=False)(right); left = ShiftRight(shiftcount=shiftcount, isLeft=True)(left)
    
    rconv4_3,rconv2_16 = sharedCNN(right,[conv0_1,conv0_2,conv0_3],[conv1x,conv2x,conv3x,conv4x])
    r_spp = SPP(rconv4_3,rconv2_16,branch_1,branch_2,branch_3,branch_4,fusion1,fusion2,H=left_input.shape[1],W=left_input.shape[2])
    lconv4_3,lconv2_16 = sharedCNN(left,[conv0_1,conv0_2,conv0_3],[conv1x,conv2x,conv3x,conv4x])
    l_spp = SPP(lconv4_3,lconv2_16,branch_1,branch_2,branch_3,branch_4,fusion1,fusion2,H=left_input.shape[1],W=left_input.shape[2])

    costvolume_i = cost_volume(l_spp,r_spp)
    costvolume_i = layers.Reshape((int(1),int(costvolume_i.shape[1]),int(costvolume_i.shape[2]),int(costvolume_i.shape[3])))(costvolume_i)
    costvolume_arr.append(costvolume_i)
  
  costvolume = layers.Concatenate(axis=1)(costvolume_arr)

  ##Model Output and Make Model
  if basic3DCNN == True:
    output = CNN3D_Basic(costvolume,filter_count=base_filter_count)
    return Model(inputs=[left_input,right_input],outputs=[output])
  else:
    output1,output2,output3 = CNN3D(costvolume,filter_count=base_filter_count)
    return Model(inputs=[left_input,right_input],outputs=[output1,output2,output3])

def smoothL1(y_true, y_pred, HUBER_DELTA = 1.0):
   x   = K.abs(y_true - y_pred)
   x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
   return  K.sum(x)