#Torch Stuff
import torch
from torch import nn

#Keras Stuff
import tensorflow as tf
from tensorflow.keras import layers

class MyNet(nn.Module):
	def __init__(self,num_classes):
		super(MyNet, self).__init__()

		def forward(self, x):
        	return self.pipe(x)

if __name__ == '__main__':
	net = MyNet(num_classes=10)

	print('\nSmt been done')