import sys
import os
import inspect

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

import tensorflow as tf
from data_loader.DataPrepare import DataProcess
from models.Model import CNN
from trainers.trainer import Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def run():
	# capture the config path from the run arguments
	# then process the json configuration file
	try:
		args = get_args()
		config = process_config(args.config)

	except ValueError:
		print("missing or invalid arguments")
		exit(0)

	# create the experiments dirs
	create_dirs([config.summary_dir, config.checkpoint_dir])

	data = DataProcess(config)
	# create tensorflow session
	sess = tf.Session()
	# create an instance of the model you want
	model = CNN(config)

	#   load model if exists
	# model.load(sess)

	# create your data generator

	# data = DataGenerator(config)
	#   create tensorboard logger
	logger = Logger(sess, config)

	#   create trainer and pass all the previous components to it
	trainer = Trainer(sess, model, data, config, logger)

	#   here you train your model
	trainer.train()


if __name__ == '__main__':
	run()
	print("Done")
