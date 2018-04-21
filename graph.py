import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

loss1 = np.load('lstm1_loss.npy')
loss2 = np.load('lstm2_loss.npy')
loss3 = np.load('lstm3_loss.npy')

iteration = xrange(1, 11)
plt.plot(iteration, loss1, iteration, loss2, iteration, loss3)
plt.legend(['LSTM (1)' , 'LSTM (2)', 'LSTM (3)'], loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy Loss')
plt.title('Entropy Loss of LSTMs on one sequence')
plt.show()
plt.savefig('results.png')
