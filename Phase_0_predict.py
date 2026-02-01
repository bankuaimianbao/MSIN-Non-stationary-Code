import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import CrossEntropyLoss
import time
from model import Transformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.nn import CosineSimilarity
from utils import select_examplers

from trainer import Trainer
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#设置设备
#下载数据
#data_all_feature = np.load("order_data/order_cut_FFT_2560_2_nor.npy")
#data_all_label = np.load("order_data/label_order_2_5832.npy")
data_all_feature = np.load("/home/djhpc/PycharmProjects/Zhang_continual/MCC5_THU_dataset_detection/FFT_and_nor/stage_0_FFT_2560_nor.npy")
data_all_label = np.load("/home/djhpc/PycharmProjects/Zhang_continual/MCC5_THU_dataset_detection/label_FFT_2560.npy")
features = data_all_feature
labels_init = data_all_label
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
labels_2d = labels_init.reshape(-1, 1)
targets = encoder.fit_transform(labels_2d)

random_seed = 42
features_train, features_valid, labels_train, labels_valid = train_test_split(features,
                                                                              targets,
                                                                              test_size=0.27, random_state=random_seed,
                                                                              shuffle=True
                                                                              )

bs = 36
np.save(f"stage_0_feature_test_nor.npy", features_valid)
np.save(f"stage_0_label_test_nor.npy", labels_valid)

features_train = torch.tensor(features_train, dtype=torch.float32)
features_valid = torch.tensor(features_valid, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.float32)
labels_valid = torch.tensor(labels_valid, dtype=torch.float32)

dataset_train = TensorDataset(features_train, labels_train)  # 将上面的特征值与标签值生成一个可迭代的数据集对象
data_loader_train = DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=0)

dataset_valid = TensorDataset(features_valid, labels_valid)  # 将上面的特征值与标签值生成一个可迭代的数据集对象
data_loader_valid = DataLoader(dataset_valid, batch_size=bs, shuffle=True, num_workers=0)
#至此data_loader_train和data_loader_valid都是一个可以按batch依次输出数据的数据集

model = Transformer()
optimizer = optim.SGD(model.parameters(),lr=1e-3,momentum=0.99)
criterion = CrossEntropyLoss()
cos_sim = CosineSimilarity(dim=1, eps=1e-6)
softmax = nn.Softmax(dim=1)
trainer = Trainer(model,optimizer,criterion,device)
loss_train = trainer.train_initial_stage(data_loader_train,epochs=80)

loss_valid = []
target_eval = []
out_eval = []

#Test
model.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader_valid):
        inputs = inputs.to(device)
        enc_outputs_1_1,enc_outputs_1_2,enc_outputs_2_1,enc_outputs_2_2,enc_outputs_3_1,enc_outputs_3_2,enc_outputs_4_1,enc_outputs_4_2,enc_outputs_5_1,enc_outputs_5_2,enc_outputs_6_1,enc_outputs_6_2, enc_outputs_7,outputs, enc_self_attns = model(inputs)

        predicted_indices = torch.argmax(outputs, dim=1)
        targets = targets.to(device)
        true_indices = torch.argmax(targets, dim=1)

        for j in range(36):
            out_eval.append(predicted_indices[j].item())
            target_eval.append(true_indices[j].item())

        loss = criterion(outputs, targets)
        loss_valid.append(loss.item())

# print(len(target_1))
# print(len(out_1))

#保存参数
PATH = 'stage_0_FFT_2560_nor_0_parameters'
torch.save(model.state_dict(),PATH)

accuracy = accuracy_score(target_eval, out_eval)
print("accuracy:", accuracy)
selected_inputs, selected_labels = select_examplers(model,data_loader_train, 0,device)

plt.plot(loss_train, label='loss_train', color='blue')
plt.title('loss_train')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.show()
