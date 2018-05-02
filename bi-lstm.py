import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from utility import *

# load data
data = create_dataset(0)
split = int(len(data)*0.80)
train, test = data[:split], data[split:]
print "Dataset created"

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.lstm = nn.LSTM(22,9)
		self.lstm2 = nn.LSTM(22,9)
                self.linear = nn.Linear(9,9)
                self.hidden = self.init_hidden()
                self.hidden2 = self.init_hidden()

	def init_hidden(self):
		return (autograd.Variable(torch.randn(1, 1, 9)),
			autograd.Variable(torch.randn((1, 1, 9))))

	def forward(self,sequence):
                inputs = [Variable(torch.Tensor(x)) for x in sequence]
                straight, back = [], []
                self.hidden = self.init_hidden()
                self.hidden2 = self.init_hidden()
                for index in xrange(len(inputs)):
                    out, self.hidden = self.lstm(inputs[index].view(1,1,-1), self.hidden)
                    straight.append(out)
                for index in xrange(len(inputs) - 1, -1, -1):
                    out2, self.hidden2 = self.lstm2(inputs[index].view(1,1,-1), self.hidden2)
                    back.append(out2)
                y = []
                for i,j in zip(straight,back):
                    z = i + j
		    prediction = self.linear(z)
                    prediction = F.relu(prediction)
                    y.append(prediction)
		return y

model = Model()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss = 0

loss_array = []

for epoch in xrange(3):
	# Note: reset loss such that doesn't accumulate after each epoch
	for sequence in xrange(len(train)):
		#inputs = [Variable(torch.Tensor(x)) for x in train[sequence][0]]
		outputs = [Variable(torch.Tensor(y)).view(1,9).long() for y in train[sequence][1]]		
		loss = 0
		optimizer.zero_grad()
		y_pred = model(train[sequence][0])
                for pred,label in zip(y_pred,outputs):
			loss += loss_function(pred.view(1,9), torch.max(label, 1)[1])
		l = loss[0].data.numpy().tolist()
                loss_array.append(l)
		print 'Sequence ', (sequence + 1), l
		loss.backward()
		optimizer.step()

np.save('bi-lstm_loss.npy',loss_array)
print 'Done'
torch.save(model.state_dict(), "bi-lstm.model")
