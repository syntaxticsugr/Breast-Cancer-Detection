
import numpy as np
from keras.callbacks import LearningRateScheduler



class CosineDecayWithWarmup(LearningRateScheduler):
    def __init__(self, learning_rate_base, total_epochs, warmup_epochs, verbose=0):
        super(CosineDecayWithWarmup, self).__init__(self.lr_schedule, verbose)
        self.learning_rate_base = learning_rate_base
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

    def lr_schedule(self, epoch):
        if epoch < self.warmup_epochs:
            lr = (self.learning_rate_base / self.warmup_epochs) * (epoch + 1)
        else:
            lr = self.learning_rate_base * 0.5 * (1 + np.cos((epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs) * np.pi))
        return lr
