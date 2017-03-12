import numpy as np
import matplotlib.pyplot as plt

greyhounds = labradors = 500

greyhound_heights = 28 + (4 * np.random.randn(greyhounds))
labrador_heights = 24 + (4 * np.random.randn(labradors))

plt.hist([greyhound_heights, labrador_heights], stacked=True, color=['r', 'b'])
plt.show()
