import matplotlib.pyplot as plt


plt.plot([6.36092,6.365955,6.381085,6.381025],'ro',label = 'All features')
plt.plot([6.8665,6.8665,6.8665,6.8665],'bo',label = 'Only PM2.5')
plt.legend()
plt.xlabel('Regularization factor (10^-N)')
plt.ylabel('Loss')
# plt.show()
plt.savefig('Plot')