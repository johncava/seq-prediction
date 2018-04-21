import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utility import *

# load data
data = create_dataset()
print "Dataset created"
lstm = nn.LSTM(22, 9)  # Input dim is 3, output dim is 3

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
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm.parameters(), lr=1e-3)

# initialize the hidden state. Keep hidden layer resets out of the training phase (maybe except when testing)
hidden = (autograd.Variable(torch.randn(1, 1, 9)),
         autograd.Variable(torch.randn((1, 1, 9))))

inputs = [Variable(torch.Tensor(x)) for x in data[0][0]]
#outputs = [Variable(torch.Tensor(y)).view(1,9) for y in data[0][1]]
outputs = [Variable(torch.Tensor(y)).view(1,9).long() for y in data[0][1]]

loss = 0

for epoch in xrange(10):
	#l = 0
	# Note: reset loss such that doesn't accumulate after each epoch
	loss = 0
	optimizer.zero_grad()
	for i, label in zip(inputs,outputs):
		# Step through the sequence one element at a time.
		# after each step, hidden contains the hidden state.
		out, hidden = lstm(i.view(1, 1, -1), hidden)
		#loss += loss_function(out.view(1,9),label)
		loss += loss_function(out.view(1,9), torch.max(label, 1)[1])
		#l = loss
	print loss[0].data.numpy().tolist()[0]
	loss.backward(retain_graph=True)
	optimizer.step()

