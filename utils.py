import torch
import numpy as np
import torch.utils.data as Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from model import Transformer
from sklearn.metrics import accuracy_score
from torch.nn import CosineSimilarity
import random
from torch.nn.modules.loss import CrossEntropyLoss
# 自定义的批次生成代码
def generate_batches(dataset1, dataset2, batch_size=36, num_batches=300, samples_per_batch_from_dataset1=12):
    # 计算总共需要从数据集1中抽取的样本数量
    total_samples_from_dataset1 = samples_per_batch_from_dataset1 * num_batches

    if len(dataset1) < total_samples_from_dataset1:
        raise ValueError("数据集1中的样本数量不足以满足要求")

    # 随机打乱数据集1的索引，并选择前 `total_samples_from_dataset1` 个索引
    indices_dataset1 = list(range(len(dataset1)))
    random.shuffle(indices_dataset1)
    selected_indices_dataset1 = indices_dataset1[:total_samples_from_dataset1]

    batches = []
    for i in range(num_batches):
        # 从数据集1中抽取18个不放回的样本及其标签
        batch_indices_dataset1 = selected_indices_dataset1[
                                 i * samples_per_batch_from_dataset1: (i + 1) * samples_per_batch_from_dataset1]
        batch_data1 = [dataset1[j][0] for j in batch_indices_dataset1]
        batch_labels1 = [dataset1[j][1] for j in batch_indices_dataset1]

        # 从数据集2中有放回地随机抽取18个样本及其标签
        batch_indices_dataset2 = random.choices(list(range(len(dataset2))), k=36 - samples_per_batch_from_dataset1)
        batch_data2 = [dataset2[j][0] for j in batch_indices_dataset2]
        batch_labels2 = [dataset2[j][1] for j in batch_indices_dataset2]

        # 合并两个部分成一个批次
        combined_batch_data = batch_data1 + batch_data2
        combined_batch_labels = batch_labels1 + batch_labels2

        combined_batch_data = torch.stack(combined_batch_data)
        combined_batch_labels = torch.stack(combined_batch_labels)

        batches.append((combined_batch_data, combined_batch_labels))

    return batches

def generate_batches_2(dataset1, dataset2, dataset3, batch_size=36, num_batches=200, samples_per_batch_from_dataset1=24,samples_per_batch_from_dataset2=6):
    # 计算总共需要从数据集1中抽取的样本数量
    total_samples_from_dataset1 = samples_per_batch_from_dataset1 * num_batches

    if len(dataset1) < total_samples_from_dataset1:
        raise ValueError("数据集1中的样本数量不足以满足要求")

    # 随机打乱数据集1的索引，并选择前 `total_samples_from_dataset1` 个索引
    indices_dataset1 = list(range(len(dataset1)))
    random.shuffle(indices_dataset1)
    selected_indices_dataset1 = indices_dataset1[:total_samples_from_dataset1]

    batches = []
    for i in range(num_batches):
        # 从数据集1中抽取24个不放回的样本及其标签
        batch_indices_dataset1 = selected_indices_dataset1[i * samples_per_batch_from_dataset1 : (i + 1) * samples_per_batch_from_dataset1]
        batch_data1 = [dataset1[j][0] for j in batch_indices_dataset1]
        batch_labels1 = [dataset1[j][1] for j in batch_indices_dataset1]

        # 从数据集2中有放回地随机抽取6个样本及其标签
        batch_indices_dataset2 = random.choices(list(range(len(dataset2))), k=samples_per_batch_from_dataset2)
        batch_data2 = [dataset2[j][0] for j in batch_indices_dataset2]
        batch_labels2 = [dataset2[j][1] for j in batch_indices_dataset2]

        # 从数据集3中有放回地随机抽取6个样本及其标签
        batch_indices_dataset3 = random.choices(list(range(len(dataset3))), k=batch_size - samples_per_batch_from_dataset1 - samples_per_batch_from_dataset2)
        batch_data3 = [dataset3[j][0] for j in batch_indices_dataset3]
        batch_labels3 = [dataset3[j][1] for j in batch_indices_dataset3]


        # 合并两个部分成一个批次
        combined_batch_data = batch_data1 + batch_data2 + batch_data3
        combined_batch_labels = batch_labels1 + batch_labels2 + batch_labels3

        combined_batch_data = torch.stack(combined_batch_data)
        combined_batch_labels = torch.stack(combined_batch_labels)

        batches.append((combined_batch_data, combined_batch_labels))

    return batches


def select_examplers(model, data_loader_train, stage,device):
    """范例选择核心逻辑（复用原代码逻辑）"""
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)
    inputs_tensor = []
    targets_tensor = []
    feature_tensor = []
    DEVICE = device
    criterion = CrossEntropyLoss()

    BATCH_SIZE =36
    CLASSES_NUM = 6
    SAMPLES_PER_CLASS = 24
    target_eval = []
    out_eval = []
    tensor_middle = []
    feature_tensor = []
    inputs_tensor = []
    targets_tensor = []
    # 提取模型中间层特征
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader_train):
            for j in range(BATCH_SIZE):
                inputs_tensor.append(inputs[j])
                targets_tensor.append(targets[j])

            inputs = inputs.to(DEVICE)
            enc_outputs_1_1, enc_outputs_1_2, enc_outputs_2_1, enc_outputs_2_2, enc_outputs_3_1, enc_outputs_3_2, enc_outputs_4_1, enc_outputs_4_2, enc_outputs_5_1, enc_outputs_5_2, enc_outputs_6_1, enc_outputs_6_2, enc_outputs_7,outputs,dec_logits = model(
                inputs)

            enc_1 = enc_outputs_1_1.view(BATCH_SIZE, 2560)
            enc_2 = enc_outputs_1_2.view(BATCH_SIZE, 2560)
            enc_3 = enc_outputs_2_1.view(BATCH_SIZE, 2560)
            enc_4 = enc_outputs_2_2.view(BATCH_SIZE, 2560)
            enc_5 = enc_outputs_3_1.view(BATCH_SIZE, 2560)
            enc_6 = enc_outputs_3_2.view(BATCH_SIZE, 2560)
            enc_7 = enc_outputs_4_1.view(BATCH_SIZE, 2560)
            enc_8 = enc_outputs_4_2.view(BATCH_SIZE, 2560)
            enc_9 = enc_outputs_5_1.view(BATCH_SIZE, 2560)
            enc_10 = enc_outputs_5_2.view(BATCH_SIZE, 2560)
            enc_11 = enc_outputs_6_1.view(BATCH_SIZE, 2560)
            enc_12 = enc_outputs_6_2.view(BATCH_SIZE, 2560)
            tensor_middle = [enc_1, enc_2, enc_3, enc_4, enc_5, enc_6, enc_7, enc_8, enc_9, enc_10, enc_11, enc_12,
                             enc_outputs_7]
            stacked_tensors = torch.stack(tensor_middle)
            average_tensor = stacked_tensors.mean(dim=0)

            average_tensor_cpu = average_tensor.to(torch.device("cpu"))
            for j in range(BATCH_SIZE):
                feature_tensor.append(average_tensor_cpu[j])

            predicted_indices = torch.argmax(outputs, dim=1)
            targets = targets.to(DEVICE)
            true_indices = torch.argmax(targets, dim=1)

            for j in range(BATCH_SIZE):
                out_eval.append(predicted_indices[j].item())
                target_eval.append(true_indices[j].item())

            loss = criterion(outputs, targets)
    # 转换为stacked tensor
    input_stack = torch.stack(inputs_tensor, dim=0)
    target_stack = torch.stack(targets_tensor, dim=0)
    feature_stack = torch.stack(feature_tensor, dim=0)

    # 按类别分组
    label_indices = torch.argmax(target_stack, dim=1)
    sub_matrices = {i: {'inputs': [], 'features': []} for i in range(CLASSES_NUM)}
    for i, label_index in enumerate(label_indices):
        sub_matrices[label_index.item()]['inputs'].append(input_stack[i])
        sub_matrices[label_index.item()]['features'].append(feature_stack[i])
    # 转换为tensor
    for category in sub_matrices:
        sub_matrices[category]['inputs'] = torch.stack(sub_matrices[category]['inputs'])
        sub_matrices[category]['features'] = torch.stack(sub_matrices[category]['features'])

    # 迭代选择范例
    selected_inputs = []
    selected_labels = []
    for category in range(CLASSES_NUM):
        mean_feature = sub_matrices[category]['features'].mean(dim=0)
        # 选第一个样本（距离中心最近）
        distances = torch.norm(sub_matrices[category]['features'] - mean_feature, dim=1)
        nearest_idx = torch.argmin(distances).item()
        selected_features = [sub_matrices[category]['features'][nearest_idx]]
        selected_input = [sub_matrices[category]['inputs'][nearest_idx]]
        selected_labels.extend([category] * len(selected_input))
        # 迭代选剩余样本
        for _ in range(1, SAMPLES_PER_CLASS):
            current_mean = torch.stack(selected_features).mean(dim=0)
            combined_distances = []
            for feature in sub_matrices[category]['features']:
                new_mean = (current_mean * len(selected_features) + feature) / (len(selected_features) + 1)
                distance = torch.norm(new_mean - mean_feature)
                combined_distances.append(distance)
            next_idx = torch.argmin(torch.tensor(combined_distances)).item()
            selected_features.append(sub_matrices[category]['features'][next_idx])
            selected_input.append(sub_matrices[category]['inputs'][next_idx])
            selected_labels.append(category)
        selected_inputs.extend(selected_input)

    # 转换为tensor并保存
    selected_inputs = torch.stack(selected_inputs)
    selected_inputs = selected_inputs.squeeze()
    selected_labels = torch.tensor(selected_labels)
    # 保存范例
    np.save(f"stage_{stage}_FFT_2560_nor_examplers_144_features_selection.npy", selected_inputs.numpy())
    np.save(f"stage_{stage}_FFT_2560_nor_examplers_144_labels_selection.npy", selected_labels.numpy())

    return selected_inputs, selected_labels



