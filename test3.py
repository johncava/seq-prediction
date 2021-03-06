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
		self.lstm3 = nn.LSTM(9,9)
		#self.sigmoid = nn.Sigmoid()
		self.hidden = self.init_hidden()
		self.hidden2 = self.init_hidden()
		self.hidden3 = self.init_hidden()

	def init_hidden(self):
		return (autograd.Variable(torch.randn(1, 1, 9)),
				autograd.Variable(torch.randn((1, 1, 9))))

	def forward(self,i):
		out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)
		out2, self.hidden2 = self.lstm2(out.view(1,1,-1), self.hidden2)
		out3, self.hidden3 = self.lstm2(out2.view(1,1,-1), self.hidden3)
		return out3

model = Model()
model.load_state_dict(torch.load('lstm3.model'))

# Testin
average_list = []
for sequence in xrange(len(test)):
	inputs = [Variable(torch.Tensor(x)) for x in test[sequence][0]]
	output = [Variable(torch.Tensor(y)) for y in test[sequence][1]]
	model.hidden = model.init_hidden()
	model.hidden2 = model.init_hidden()
	model.hidden3 = model.init_hidden()
	accuracy = 0
	for i, label in zip(inputs, output):
		prediction = model(i).view(1,9)
		a = torch.max(prediction,1)[1].data.numpy().tolist()[0]
		b = torch.max(label,0)[1].data.numpy().tolist()
		if a == b:
			accuracy += 1
	average_list.append(accuracy/float(len(output)))	

print average_list
print "-------"
print np.average(average_list)
