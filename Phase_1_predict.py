import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.modules.loss import CrossEntropyLoss
from model import Transformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import generate_batches,select_examplers
from LabeledDataset import LabeledDataset
from trainer import Trainer
from copy import deepcopy
import torch.nn.functional as F
# from torch.nn import CosineSimilarity
# import seaborn as sns
import time
start_time = time.time()
if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置设备
# 下载数据
data_all_feature_1 = np.load("/home/djhpc/PycharmProjects/Zhang_continual/MCC5_THU_dataset_detection/FFT_and_nor/stage_1_FFT_2560_nor.npy")
data_all_label_1 = np.load("/home/djhpc/PycharmProjects/Zhang_continual/MCC5_THU_dataset_detection/label_FFT_2560.npy")
criterion = CrossEntropyLoss()

# 初始化模型
model = Transformer()
model_teach = Transformer()
# 下载模型参数
PATH = 'stage_0_FFT_2560_nor_0_parameters'
model.load_state_dict(torch.load(PATH))
model_teach.load_state_dict(torch.load(PATH))
model.to(device)
model_teach.to(device)
#将教师模型参数冻结
for param in model_teach.parameters():
    param.requires_grad = False
# 处理阶段1的数据为机器学习的标准训练格式
features_1 = np.expand_dims(data_all_feature_1, axis=1)
targets_1 = np.expand_dims(data_all_label_1, axis=1)
labels_init_1 = targets_1
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
labels_2d = labels_init_1.reshape(-1, 1)
targets_1 = encoder.fit_transform(labels_2d)
random_seed = 42

features_train_1, features_valid_1, labels_train_1, labels_valid_1 = train_test_split(features_1,
                                                                                      targets_1,
                                                                                      test_size=0.27,
                                                                                      random_state=random_seed,
                                                                                      shuffle=True
                                                                                      )
bs = 36
np.save(f"stage_1_feature_test_nor.npy", features_valid_1)
np.save(f"stage_1_label_test_nor.npy", labels_valid_1)
# 引入阶段0的144个训练数据
data_0_examples = np.load("stage_0_FFT_2560_nor_examplers_144_features_selection.npy")
label_0_examples = np.load("stage_0_FFT_2560_nor_examplers_144_labels_selection.npy")

features_train_0 = np.expand_dims(data_0_examples, axis=1)
targets_0 = np.expand_dims(label_0_examples, axis=1)
labels_init_0 = targets_0
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
labels_2d = labels_init_0.reshape(-1, 1)
labels_train_0 = encoder.fit_transform(labels_2d)
# 阶段0的测试数据集
features_valid_0 = np.load("stage_1_feature_test_nor.npy")
labels_valid_0 = np.load("stage_1_label_test_nor.npy")

# 生成可循环的训练集
features_train_0 = torch.tensor(features_train_0, dtype=torch.float32)
features_train_1 = torch.tensor(features_train_1, dtype=torch.float32)
features_valid_0 = torch.tensor(features_valid_0, dtype=torch.float32)
features_valid_1 = torch.tensor(features_valid_1, dtype=torch.float32)

labels_train_0 = torch.tensor(labels_train_0, dtype=torch.float32)
labels_train_1 = torch.tensor(labels_train_1, dtype=torch.float32)
labels_valid_0 = torch.tensor(labels_valid_0, dtype=torch.float32)
labels_valid_1 = torch.tensor(labels_valid_1, dtype=torch.float32)

feature_train_dataset_1 = LabeledDataset(features_train_1, labels_train_1)
feature_train_dataset_0 = LabeledDataset(features_train_0, labels_train_0)
dataset_train_1 = TensorDataset(features_train_1, labels_train_1)  # 将上面的特征值与标签值生成一个可迭代的数据集对象
data_loader_train_1 = DataLoader(dataset_train_1 , batch_size=bs, shuffle=True, num_workers=0)


batches = generate_batches(feature_train_dataset_1, feature_train_dataset_0, batch_size=36, num_batches=150,
                           samples_per_batch_from_dataset1=24)
# 生成可循环的测试集
dataset_valid_0 = TensorDataset(features_valid_0, labels_valid_0)  # 将上面的特征值与标签值生成一个可迭代的数据集对象
data_loader_valid_0 = DataLoader(dataset_valid_0, batch_size=bs, shuffle=True, num_workers=0)
dataset_valid_1 = TensorDataset(features_valid_1, labels_valid_1)  # 将上面的特征值与标签值生成一个可迭代的数据集对象
data_loader_valid_1 = DataLoader(dataset_valid_1, batch_size=bs, shuffle=True, num_workers=0)

optimizer = optim.SGD(model.parameters(), lr=2e-4, momentum=0.99)
softmax = nn.Softmax(dim=1)

trainer = Trainer(model,optimizer,criterion,device)

history = trainer.train_incremental_stage(
    model_teach=model_teach,
    batches=batches,
    epochs=2,
    alpha=0.9991,
    T=2
)

loss_hard_train = []
loss_ewc_train = []
loss_soft_train_1 = []
loss_soft_train_2 = []
loss_soft_train_3 = []
loss_valid_0 = []
loss_valid_1 = []

target_eval_0 = []
out_eval_0 = []
target_eval_1 = []
out_eval_1 = []

# Test
model.eval()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader_valid_0):
        # for inputs, targets in batches:
        inputs = inputs.to(device)
        enc_outputs_1_1, enc_outputs_1_2, enc_outputs_2_1, enc_outputs_2_2, enc_outputs_3_1, enc_outputs_3_2, enc_outputs_4_1, enc_outputs_4_2, enc_outputs_5_1, enc_outputs_5_2, enc_outputs_6_1, enc_outputs_6_2,enc_outputs_7, outputs, enc_self_attns = model(
            inputs)

        predicted_indices = torch.argmax(outputs, dim=1)
        targets = targets.to(device)
        true_indices = torch.argmax(targets, dim=1)

        for j in range(36):
            out_eval_0.append(predicted_indices[j].item())
            target_eval_0.append(true_indices[j].item())

        loss_0 = criterion(outputs, targets)
        loss_valid_0.append(loss_0.item())

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader_valid_1):
        # for inputs, targets in batches:
        inputs = inputs.to(device)
        enc_outputs_1_1, enc_outputs_1_2, enc_outputs_2_1, enc_outputs_2_2, enc_outputs_3_1, enc_outputs_3_2, enc_outputs_4_1, enc_outputs_4_2, enc_outputs_5_1, enc_outputs_5_2, enc_outputs_6_1, enc_outputs_6_2,enc_outputs_7, outputs, enc_self_attns = model(
            inputs)

        predicted_indices = torch.argmax(outputs, dim=1)
        targets = targets.to(device)
        true_indices = torch.argmax(targets, dim=1)

        for j in range(36):
            out_eval_1.append(predicted_indices[j].item())
            target_eval_1.append(true_indices[j].item())

        loss_1 = criterion(outputs, targets)
        loss_valid_1.append(loss_1.item())

accuracy_0 = accuracy_score(target_eval_0, out_eval_0)
print("accuracy_0:", accuracy_0)
accuracy_1 = accuracy_score(target_eval_1, out_eval_1)
print("accuracy_1:", accuracy_1)

#保存参数
PATH = 'stage_1_FFT_2560_nor_0_1_parameters'
torch.save(model.state_dict(),PATH)
selected_inputs, selected_labels = select_examplers(model,data_loader_train_1, 1,device)

plt.plot(history['loss_train'], label='loss_train', color='blue')
plt.title('loss_train')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.show()


