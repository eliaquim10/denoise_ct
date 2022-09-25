from .libs import *

LR_SIZE = 24
HR_SIZE = 96
training = True
gan_training = False
Loader_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
metric_mae = MeanAbsoluteError()

def __main__():
    pass
if __name__ == "__main__":
    pass