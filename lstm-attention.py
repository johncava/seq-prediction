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
		self.sigmoid = nn.Sigmoid()
		self.hidden = self.init_hidden()
		self.previous = []

	def init_hidden(self):
		return (autograd.Variable(torch.zeros(1, 1, 9)),
			autograd.Variable(torch.zeros((1, 1, 9))))
	
	def init_previous(self):
		return []

	def forward(self,i):
		out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)
		final_out = out
		if len(self.previous) > 0:
			attention = []
			for v in self.previous:
				attention.append(F.tanh(v + out))
			attention = torch.cat(attention)
			attention = F.softmax(attention)
			a = []
			for i in attention:
				a.append(out*i)
			a = torch.cat(a)
			a = torch.mean(a,dim=0).view(1,9)
			final_out = final_out + a 			
		self.previous.append(out)
		return final_out

model = Model()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_array = []

for epoch in xrange(3):
	#l = 0
	# Note: reset loss such that doesn't accumulate after each epoch
	for sequence in xrange(5):#len(train)):
		inputs = [Variable(torch.Tensor(x)) for x in train[sequence][0]]
		outputs = [Variable(torch.Tensor(y)).view(1,9).long() for y in train[sequence][1]]		
		loss = 0
		optimizer.zero_grad()
		model.hidden = model.init_hidden()	
		model.previous = model.init_previous()
		for i, label in zip(inputs,outputs):
			# Step through the sequence one element at a time.
			# after each step, hidden contains the hidden state.
			out = model(i)
			#loss += loss_function(out.view(1,9),label)
			loss += loss_function(out.view(1,9), torch.max(label, 1)[1])
			#l = loss
		#loss_array.append(loss[0].data.numpy().tolist()[0])
		print 'Sequence ', (sequence + 1)
		loss.backward()#retain_graph=True)
		optimizer.step()
	break
#np.save('lstm1_loss.npy',loss_array)
print 'Done '
#torch.save(model.state_dict(), "lstm1.model")
