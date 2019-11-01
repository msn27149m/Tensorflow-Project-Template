import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold


class DataProcess(object):
	def __init__(self, configs):
		self.config = configs
		# load data here
		data = fetch_openml("mnist_784")
		self.X = data.data
		self.y = data.target
		self.X_fac = None
		self.y_fac = None
		self.X_val = None
		self.y_val = None
		self.kFold = None
		self.kFoldSplit = None
		self.run_valid_splitter()

	def run_valid_splitter(self):
		self.X_fac, self.X_val, self.y_fac, self.y_val = train_test_split(self.X, self.y, test_size=0.2)
		self._RunKFold()

	def _InitKFold(self):
		if self.config.KFoldNme.lower() == "stratifiedkfold":
			self.kFold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
		elif self.config.KFoldNme.lower() == "reatedstratifiedkfold":
			self.kFold = RepeatedStratifiedKFold(n_splits=self.config.num_iter_per_epoch, n_repeats=2)

	def _RunKFold(self):
		self._InitKFold()
		self.kFoldSplit = self.kFold.split(self.X_fac, self.y_fac)

	def next_batch(self):
		try:
			idx_train, idx_test = next(self.kFoldSplit)
			return self.X_fac[idx_train], self.y_fac[idx_train], self.X_fac[idx_test], self.y_fac[idx_test]
		except StopIteration as err:
			print("StopIteration")
		finally:
			self._RunKFold()


if __name__ =="__main__":
	config = {}
	data = DataGenerator(config)
	for n in range(0, 2):
		t = next(data.next_batch())
		print(t)
