import io

import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from model import BertClassifier
from constants import *
from dataModule import SequenceDataset
from preprocessor import Preprocessor
from utils import seed_everything
from datetime import datetime
import os

seed_everything(24)




def train(data):
    # load dutch tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

    preprocessor = Preprocessor()
    train_dataset = SequenceDataset(data, tokenizer, preprocessor)

    # model configuration
    config = {'hidden_size': 768,
              'num_labels': train_dataset.label_count,
              'hidden_dropout_prob': 0.05,
              }

    # Create our custom BERTClassifier model object
    model = BertClassifier(config)
    model.id_dict = train_dataset.id_dict
    model.to(DEVICE)

    validation_split = 0.2
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True

    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

    print('Training Set Size {}, Validation Set Size {}'.format(len(train_indices), len(val_indices)))

    loss_fn = nn.CrossEntropyLoss(weight=Tensor(train_dataset.class_weights).to(DEVICE))

    optimizer = Adam([
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 3e-4}
    ])

    model.zero_grad()
    training_acc_list, validation_acc_list = [], []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        train_correct_total = 0
        # Training Loop
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            model.train(True)

            inputs = batch[0]
            labels = batch[1].to(DEVICE)

            logits = model(**inputs)

            loss = loss_fn(logits, labels)

            loss.backward()

            epoch_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            _, predicted = torch.max(logits.data, 1)
            correct_reviews_in_batch = (predicted == labels).sum().item()
            train_correct_total += correct_reviews_in_batch

        print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

        # Validation Loop
        with torch.no_grad():
            val_correct_total = 0

            model.train(False)
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                inputs = batch[0]

                labels = batch[1].to(DEVICE)
                logits = model(**inputs)

                _, predicted = torch.max(logits.data, 1)
                correct_reviews_in_batch = (predicted == labels).sum().item()
                val_correct_total += correct_reviews_in_batch

            training_acc_list.append(train_correct_total * 100 / len(train_indices))
            validation_acc_list.append(val_correct_total * 100 / len(val_indices))
            print('Training Accuracy {:.4f} - Validation Accurracy {:.4f}'.format(
                train_correct_total * 100 / len(train_indices), val_correct_total * 100 / len(val_indices)))

        torch.save(model, MODEL_FILE_PATH[:-4] + f"_epoch-{epoch}" + ".pth")

    torch.save(model, MODEL_FILE_PATH)

    return MODEL_FILE_PATH
