import torch
import torch.nn.functional as F
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        """
        Args:
            model: The student transformer model to train.
            optimizer: The optimizer (e.g., SGD).
            criterion: The loss function (e.g., CrossEntropyLoss).
            device: torch.device (cpu or cuda).
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_initial_stage(self, data_loader, epochs=200):
        """
        Executes the initial training loop (Stage 0).
        Matches the logic from stage_0_transformer_encoder_predict.py.
        """
        self.model.to(self.device)
        loss_train = []

        print("Starting Initial Training Stage...")

        for epoch in range(epochs):
            print('Epoch', '%04d' % (epoch + 1))
            run_loss = 0
            self.model.train()

            for batch_idx, (inputs, targets) in enumerate(data_loader):
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                # Model returns 15 values. Index 13 is 'output' (softmax), Index 14 is 'dec_logits'
                # User Stage 0 code uses 'out' (Index 13) for loss.
                results = self.model(inputs)
                out = results[13]

                loss = self.criterion(out, targets)
                run_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            epoch_loss = run_loss / len(data_loader)
            print(epoch_loss)
            loss_train.append(epoch_loss)

        return loss_train

    def train_incremental_stage(self, model_teach, batches, epochs=200, alpha=0.9991, T=2):
        """
        Executes the incremental training loop (Stage 1 & Stage 2).
        Matches the logic from stage_1 and stage_2 scripts using Knowledge Distillation (LwF).
        """
        self.model.to(self.device)
        model_teach.to(self.device)
        model_teach.eval()

        # Freeze the teacher model
        for param in model_teach.parameters():
            param.requires_grad = False

        loss_train = []
        loss_hard_train = []
        loss_soft_train_1 = []
        loss_soft_train_2 = []
        loss_soft_train_3 = []

        print("Starting Incremental Training Stage...")

        for epoch in range(epochs):
            run_loss = 0
            run_loss_hard = 0
            run_loss_soft_1 = 0
            run_loss_soft_2 = 0
            run_loss_soft_3 = 0

            self.model.train()
            print(epoch)

            # Note: 'batches' is a pre-generated list of (inputs, targets)
            for inputs, targets in batches:
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Student Forward Pass
                # indices: 12=enc_outputs_7, 13=out(softmax), 14=dec_logits
                results_student = self.model(inputs)
                enc_outputs_7 = results_student[12]
                out = results_student[13]
                dec_logits = results_student[14]

                # Teacher Forward Pass
                results_teach = model_teach(inputs)
                enc_outputs_7_teach = results_teach[12]
                dec_logits_teach = results_teach[14]

                # Cross Projections
                teacher_cross_logits = model_teach.projection(enc_outputs_7)
                student_cross_logits = self.model.projection(enc_outputs_7_teach)

                # --- Loss Calculation ---

                # 1. Teacher Distribution
                outputs_T = F.softmax(dec_logits_teach / T, dim=1)

                # 2. Soft Loss 1: Student Features -> Teacher Classifier
                outputs_cross_1 = F.softmax(teacher_cross_logits / T, dim=1)
                loss_soft_1 = outputs_T.mul(-1 * torch.log(outputs_cross_1))
                loss_soft_1 = loss_soft_1.sum(1).mean() * (T * T)

                # 3. Soft Loss 2: Teacher Features -> Student Classifier
                outputs_cross_2 = F.softmax(student_cross_logits / T, dim=1)
                loss_soft_2 = outputs_T.mul(-1 * torch.log(outputs_cross_2))
                loss_soft_2 = loss_soft_2.sum(1).mean() * (T * T)

                # 4. Soft Loss 3: Student Distribution (Standard KD)
                outputs_S = F.softmax(dec_logits / T, dim=1)
                loss_soft_3 = outputs_T.mul(-1 * torch.log(outputs_S))
                loss_soft_3 = loss_soft_3.sum(1).mean() * (T * T)

                # 5. Hard Loss
                loss_hard = self.criterion(out, targets)

                # Total Loss
                loss = (alpha * loss_hard +
                        (1 - alpha) / 3 * loss_soft_1 +
                        (1 - alpha) / 3 * loss_soft_2 +
                        (1 - alpha) / 3 * loss_soft_3)

                run_loss += loss.item()
                run_loss_hard += loss_hard.item()
                run_loss_soft_1 += loss_soft_1.item()
                run_loss_soft_2 += loss_soft_2.item()
                run_loss_soft_3 += loss_soft_3.item()

                loss.backward()
                self.optimizer.step()

            # Logging
            epoch_loss = run_loss / len(batches)
            epoch_loss_hard = run_loss_hard / len(batches)
            epoch_loss_soft_1 = run_loss_soft_1 / len(batches)
            epoch_loss_soft_2 = run_loss_soft_2 / len(batches)
            epoch_loss_soft_3 = run_loss_soft_3 / len(batches)

            print('total loss:', epoch_loss)
            print('loss soft_1:', epoch_loss_soft_1)
            print('loss soft_2:', epoch_loss_soft_2)
            print('loss soft_3:', epoch_loss_soft_3)

            loss_train.append(epoch_loss)
            loss_hard_train.append(epoch_loss_hard)
            loss_soft_train_1.append(epoch_loss_soft_1)
            loss_soft_train_2.append(epoch_loss_soft_2)
            loss_soft_train_3.append(epoch_loss_soft_3)

        return {
            'loss_train': loss_train,
            'loss_hard': loss_hard_train,
            'loss_soft_1': loss_soft_train_1,
            'loss_soft_2': loss_soft_train_2,
            'loss_soft_3': loss_soft_train_3
        }