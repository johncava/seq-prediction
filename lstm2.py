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
split = int(len(data)*0.80)
train, test = data[:split], data[split:]
print "Dataset created"

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.lstm = nn.LSTM(22,9)
		self.lstm2 = nn.LSTM(9,9)
		#self.sigmoid = nn.Sigmoid()
		self.hidden = self.init_hidden()
		self.hidden2 = self.init_hidden()

	def init_hidden(self):
		return (autograd.Variable(torch.randn(1, 1, 9)),
				autograd.Variable(torch.randn((1, 1, 9))))

	def forward(self,i):
		out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)
		out2, self.hidden2 = self.lstm2(out.view(1,1,-1), self.hidden2)
		return out2

model = Model()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# initialize the hidden state. Keep hidden layer resets out of the training phase (maybe except when testing)
hidden = (autograd.Variable(torch.randn(1, 1, 9)),
         autograd.Variable(torch.randn((1, 1, 9))))
		 
hidden2 = (autograd.Variable(torch.randn(1, 1, 9)),
         autograd.Variable(torch.randn((1, 1, 9))))

#inputs = [Variable(torch.Tensor(x)) for x in data[0][0]]
#outputs = [Variable(torch.Tensor(y)).view(1,9).long() for y in data[0][1]]

loss = 0
loss_array = []
for epoch in xrange(3):
	# Note: reset loss such that doesn't accumulate after each epoch
	for sequence in xrange(len(train)):
		inputs = [Variable(torch.Tensor(x)) for x in train[sequence][0]]
		outputs = [Variable(torch.Tensor(y)).view(1,9).long() for y in train[sequence][1]]	
		loss = 0
		optimizer.zero_grad()
		model.hidden = model.init_hidden()
		model.hidden2 = model.init_hidden()
		for i, label in zip(inputs,outputs):
			# Step through the sequence one element at a time.
			# after each step, hidden contains the hidden state.
			out = model(i)
			loss += loss_function(out.view(1,9), torch.max(label, 1)[1])
		loss_array.append(loss[0].data.numpy().tolist()[0])
		#print 'Sequence ', (sequence + 1)
		loss.backward()#retain_graph=True)
		optimizer.step()

np.save('lstm2_loss.npy',loss_array)
print 'Done 2'
torch.save(model.state_dict(), "lstm2.model")
