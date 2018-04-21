import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from utility import *

# load data
data = create_dataset()
print "Dataset created"

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.lstm = nn.LSTM(22,9)
		self.sigmoid = nn.Sigmoid()

	def forward(self,i, hidden):
		out, hidden = self.lstm(i.view(1, 1, -1), hidden)
		return out, hidden

#lstm = nn.LSTM(22, 9)  # Input dim is 22, output dim is 9

'''
inputs = [autograd.Variable(torch.randn((1,3)))
          for _ in range(5)]  # make a sequence of length 5
data_y = [[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1]]
outputs =[autograd.Variable(torch.Tensor(y)).view(1,1,3)
	  for y in data_y]
'''
#print inputs
#print outputs
#loss_function = nn.MSELoss()
model = Model()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# initialize the hidden state. Keep hidden layer resets out of the training phase (maybe except when testing)
hidden = (autograd.Variable(torch.randn(1, 1, 9)),
         autograd.Variable(torch.randn((1, 1, 9))))

inputs = [Variable(torch.Tensor(x)) for x in data[0][0]]
#outputs = [Variable(torch.Tensor(y)).view(1,9) for y in data[0][1]]
outputs = [Variable(torch.Tensor(y)).view(1,9).long() for y in data[0][1]]

loss = 0

loss_array = []

for epoch in xrange(10):
	#l = 0
	# Note: reset loss such that doesn't accumulate after each epoch
	loss = 0
	optimizer.zero_grad()
	for i, label in zip(inputs,outputs):
		# Step through the sequence one element at a time.
		# after each step, hidden contains the hidden state.
		out, hidden = model(i, hidden)
		#loss += loss_function(out.view(1,9),label)
		loss += loss_function(out.view(1,9), torch.max(label, 1)[1])
		#l = loss
	loss_array.append(loss[0].data.numpy().tolist()[0])
	loss.backward(retain_graph=True)
	optimizer.step()

np.save('lstm1_loss.npy',loss_array)


