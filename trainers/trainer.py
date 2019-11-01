from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(Trainer, self).__init__(sess, model, data, config, logger)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)

    def train_epoch(self, epoch_n):
        loop = tqdm(range(self.config.num_iter_per_epoch))

        train_losses = []
        train_accs = []

        for _ in loop:
            self.get_batch()
            self.train_step()
            train_loss, train_acc = self.train_step()
            train_losses.append(train_loss)
            train_accs.append(train_acc)

        test_loss, test_acc = self.test_step()
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)

        summaries_dict = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        self.logger.summarize(epoch_n, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        # print(xTrain, yTrain)
        # train_x, train_y, test_x, test_y = next(self.data.next_batch(self.config.kfolds))
        train_feed_dict = {self.model.x: self.X_train, self.model.y: self.y_train, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy],
                                     feed_dict=train_feed_dict)
        return loss, acc

    def test_step(self):
        train_feed_dict = {self.model.x: self.X_test, self.model.y: self.y_test, self.model.is_training: False}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy],
                                     feed_dict=train_feed_dict)
        return loss, acc

    def get_batch(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.data.next_batch()
