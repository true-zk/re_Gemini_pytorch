import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
import torchmetrics
from model import Gemimi

# def global para
best_model_para = None

default_para_dic = {
	'embedding_size': 64,
	'Gemini_feature_size': 9,
	'embedding_depth': 2,
	'Iter': 5,
	'Epoch': 64,
	'batch': 5,
	'learning_rate': 1e-4,
	'save_path': r"model\best_model0.para"}


def valid(model, dataset_val, batch_size, device, best_val_acc):
	val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
	acc_epoch = torchmetrics.Accuracy("binary")
	auc_epoch = torchmetrics.AUROC("binary")
	loss_sum, loss_count = 0, 0

	for g0, g1, y in val_loader:
		loss, sim = model(g0.to(device), g1.to(device), y.to(device))
		loss_sum, loss_count = loss_sum + loss, loss_count + 1

		y_pred = (sim + 1) / 2
		y = (y + 1) / 2
		acc_epoch.update(y_pred, y)
		auc_epoch.update(y_pred, y)

	acc = acc_epoch.compute()
	auc = auc_epoch.compute()

	if acc > best_val_acc:
		global best_model_para
		best_model_para = model.state_dict()

	return loss_sum/loss_count, acc, auc


def train(dataset_train: Dataset, dataset_val, batch_size=default_para_dic['batch'], Epoch=default_para_dic['Epoch']):
	train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = Gemimi(
		embedding_size=default_para_dic['embedding_size'],
		Gemini_feature_size=default_para_dic['Gemini_feature_size'],
		embedding_depth=default_para_dic['embedding_depth'],
		Iter=default_para_dic['Iter'])

	optimizer = torch.optim.Adam(model.parameters(), lr=default_para_dic['learning_rate'])

	train_loss_list, train_acc_list, train_auc_list = [], [], []
	val_loss_list, val_acc_list, val_auc_list = [], [], []

	acc_epoch = torchmetrics.Accuracy("binary")
	auc_epoch = torchmetrics.AUROC("binary")

	best_val_acc = 0

	for epoch in range(Epoch):
		time_start = time.time()
		loss_sum, loss_count = 0, 0

		for g0, g1, y in train_loader:
			loss, sim = model(g0.to(device), g1.to(device), y.to(device))
			loss_sum, loss_count = loss_sum + loss, loss_count + 1

			optimizer.zero_grad()
			loss.backword()
			optimizer.step()

			y_pred = (sim + 1) / 2
			y = (y + 1) / 2
			acc_epoch.update(y_pred, y)
			auc_epoch.update(y_pred, y)

		avg_loss = loss_sum / loss_count
		acc = acc_epoch.compute()
		auc = auc_epoch.compute()
		acc_epoch.reset()
		auc_epoch.reset()

		print("Epoch {:03d}: trainTimespan:{:.2f} avg_Loss:{:.3f}, Accuracy:{:.3%}, AUC:{:.3f}".format(
			epoch, time.time()-time_start, avg_loss, acc, auc
		))

		val_loss, val_acc, val_auc = valid(model, dataset_val, batch_size, device, best_val_acc)
		print("validation: avg_Loss:{:.3f}, Accuracy:{:.3%}, AUC:{:.3f}".format(
			val_loss, val_acc, val_auc
		))

		train_acc_list.append(acc)
		train_auc_list.append(auc)
		train_loss_list.append(avg_loss)
		val_loss_list.append(val_loss)
		val_acc_list.append(val_acc)
		val_auc_list.append(val_auc)


def save():
	global best_model_para
	if best_model_para is not None:
		if os.path.exists(r"model\best_model0.para"):
			suffix = 0
			for f in os.listdir(r'model'):
				suffix_ = int(f.split('.')[0][10:])
				if suffix_ > suffix:
					suffix = suffix_
			suffix += 1
			suffix = str(suffix)
			torch.save(best_model_para, r"model\best_model" + suffix + ".para")
		else:
			torch.save(best_model_para, r"model\best_model0.para")