import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss1 = np.load('lstm1_loss.npy')
loss2 = np.load('lstm2_loss.npy')
loss3 = np.load('lstm3_loss.npy')

#plt.plot(xrange(1, len(loss1) + 1), loss1, xrange(1, len(loss2) + 1), loss2, xrange(1, len(loss3) + 1), loss3)
#plt.legend(['LSTM (1)' , 'LSTM (2)', 'LSTM (3)'], loc='upper left')
plt.plot(xrange(1,len(loss3) + 1), loss3)
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.title('Entropy Loss of LSTM (3)')
plt.show()
plt.savefig('result_lstm3.png')
