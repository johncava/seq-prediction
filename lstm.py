import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utility import *

# load data
data = create_dataset()

lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [autograd.Variable(torch.randn((1,3)))
          for _ in range(5)]  # make a sequence of length 5
data_y = [[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1]]
outputs =[autograd.Variable(torch.Tensor(y)).view(1,1,3)
	  for y in data_y]

#print inputs
#print outputs
loss_function = nn.MSELoss()
optimizer = optim.SGD(lstm.parameters(), lr=0.1)

# initialize the hidden state.
hidden = (autograd.Variable(torch.randn(1, 1, 3)),
          autograd.Variable(torch.randn((1, 1, 3))))

for epoch in xrange(10):
	l = 0
	for i, label in zip(inputs,outputs):
		# Step through the sequence one element at a time.
		# after each step, hidden contains the hidden state.
		optimizer.zero_grad()
		out, hidden = lstm(i.view(1, 1, -1), hidden)
		#print out
		loss = loss_function(out,label)
		l = loss
		loss.backward(retain_graph=True)
		optimizer.step()
	print l

