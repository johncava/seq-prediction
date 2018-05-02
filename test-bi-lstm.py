import torch
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
		return (Variable(torch.randn(1, 1, 9)),
			Variable(torch.randn((1, 1, 9))))

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
model.load_state_dict(torch.load('bi-lstm.model'))

# Testin
average_list = []
for sequence in xrange(len(test)-2):
	output = [Variable(torch.Tensor(y)) for y in test[sequence][1]]
	accuracy = 0
        prediction = model(test[sequence][0])
        for i, label in zip(prediction, output):
                #print i.view(1,9)[0]
                #print label
                a = torch.max(i.view(1,9)[0],0)[1].data.numpy().tolist()
		b = torch.max(label,0)[1].data.numpy().tolist()
		#print a
                #print b
                if a == b:
			accuracy += 1
        average_list.append(accuracy/float(len(output)))	

print average_list
print "-------"
print np.average(average_list)
