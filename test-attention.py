import torch
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
                return (Variable(torch.zeros(1, 1, 9)),
                        Variable(torch.zeros((1, 1, 9))))
        
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
model.load_state_dict(torch.load('lstm-attention.model'))
# Testin
average_list = []
for sequence in xrange(len(test)):
        inputs = [Variable(torch.Tensor(x)) for x in test[sequence][0]]
        output = [Variable(torch.Tensor(y)) for y in test[sequence][1]]
        model.hidden = model.init_hidden()
        model.previous = []
	accuracy = 0
        for i, label in zip(inputs, output):
                prediction = model(i).view(1,9)
                a = torch.max(prediction,1)[1].data.numpy().tolist()[0]
                b = torch.max(label,0)[1].data.numpy().tolist()
                #print a,b
                if a == b:
			accuracy += 1
	print sequence
        average_list.append(accuracy/float(len(output)))
print average_list
print "-------"
print np.average(average_list)
