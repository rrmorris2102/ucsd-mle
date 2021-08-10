from transformers import XLNetForSequenceClassification, XLNetTokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
from collections import defaultdict
from sklearn.metrics import classification_report

class SentimentDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
            )

        input_ids = pad_sequences(encoding['input_ids'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids) 

        attention_mask = pad_sequences(encoding['attention_mask'], maxlen=self.max_len, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask)       

        return {
        'review_text': review,
        'input_ids': input_ids,
        'attention_mask': attention_mask.flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
        }

class XLNetSentiment(object):
    """
    The main class for XLNet.
    """

    def __init__(self, model_file, batchsize=48, max_len=64):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device {}'.format(device))

        model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 3)
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)        

        self.device = device
        self.model = model

        PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
        self.tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.MAX_LEN = max_len
        self.BATCHSIZE = batchsize

        self.class_names = ['positive', 'negative', 'neutral']
        #self.class_names = ['positive', 'negative']
        
    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=self.MAX_LEN,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = pad_sequences(encoded_text['input_ids'], maxlen=self.MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
        input_ids = input_ids.astype(dtype = 'int64')
        input_ids = torch.tensor(input_ids) 

        attention_mask = pad_sequences(encoded_text['attention_mask'], maxlen=self.MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
        attention_mask = attention_mask.astype(dtype = 'int64')
        attention_mask = torch.tensor(attention_mask) 

        input_ids = input_ids.reshape(self.BATCHSIZE,self.MAX_LEN).to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        outputs = outputs[0][0].cpu().detach()

        probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
        _, prediction = torch.max(outputs, dim =-1)

        results = {
            'positive_score': probs[0],
            'negative_score': probs[1],
            'neutral_score': probs[2],
            'text': text,
            'sentiment': self.class_names[prediction]
        }

        return results

class XLNetConfig(object):
    def __init__(self):
        pass

class XLNetSentimentTrain(object):
    """
    XLNet Training
    """

    def __init__(self, batchsize=16, max_len=64):
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device {}'.format(device))

        model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 3)
        model = model.to(device)        

        self.device = device
        self.model = model

        PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
        self.tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.test_size = 0.5
        self.random_state = 101
        self.MAX_LEN = max_len
        self.BATCH_SIZE = batchsize
        self.EPOCHS = 10
        self.num_data_workers = 4
        self.model_file = './models/xlnet_model_batch{}.bin'.format(batchsize)
        self.class_names = ['positive', 'negative', 'neutral']
        #self.class_names = ['positive', 'negative']

        self.columns = None

    def train(self, df, columns):
        self.columns = columns

        df_train, df_test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        df_val, df_test = train_test_split(df_test, test_size=self.test_size, random_state=self.random_state)

        train_data_loader = self.__create_data_loader(df_train, self.MAX_LEN, self.BATCH_SIZE)
        val_data_loader = self.__create_data_loader(df_val, self.MAX_LEN, self.BATCH_SIZE)
        test_data_loader = self.__create_data_loader(df_test, self.MAX_LEN, self.BATCH_SIZE)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                                        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

        total_steps = len(train_data_loader) * self.EPOCHS

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
            )        

        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(self.EPOCHS):
            print(f'Epoch {epoch + 1}/{self.EPOCHS}')
            print('-' * 10)

            train_acc, train_loss = self.__train_epoch(
                train_data_loader,     
                optimizer, 
                scheduler, 
                len(df_train)
            )

            print(f'Train loss {train_loss} Train accuracy {train_acc}')

            val_acc, val_loss = self.__eval_model(
                val_data_loader, 
                len(df_val)
            )

            print(f'Val loss {val_loss} Val accuracy {val_acc}')
            print()

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                print('Saving model {}'.format(self.model_file))
                torch.save(self.model.state_dict(), self.model_file)
                best_accuracy = val_acc

        self.model.load_state_dict(torch.load(self.model_file))
        self.model = self.model.to(self.device)

        test_acc, test_loss = self.__eval_model(
            test_data_loader,
            len(df_test)
        )

        print('Test Accuracy :', test_acc)
        print('Test Loss :', test_loss)

        y_review_texts, y_pred, y_pred_probs, y_test = \
            self.__get_predictions(test_data_loader)

        report = classification_report(y_test, y_pred, target_names=self.class_names)

        results = {
            'classification_report': report,
            'epochs': self.EPOCHS,
            'batchsize': self.BATCH_SIZE,
            'max_len': self.MAX_LEN,
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc'],
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
        }

        return results

    def __create_data_loader(self, df, max_len, batch_size):
        ds = SentimentDataset(
            reviews=df[self.columns[0]].to_numpy(),
            targets=df[self.columns[1]].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=max_len
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.num_data_workers
        )

    def __train_epoch(self, data_loader, optimizer, scheduler, n_examples):
        model = self.model.train()
        losses = []
        acc = 0
        counter = 0
    
        for idx, d in enumerate(data_loader):
            print('train_epoch {}/{}'.format(idx, len(data_loader)))

            if len(d["input_ids"]) < self.BATCH_SIZE:
                print('Skipped partial batch (got {}, expected {})'.format(len(d["input_ids"]), self.BATCH_SIZE))
                continue

            input_ids = d["input_ids"].reshape(self.BATCH_SIZE,self.MAX_LEN).to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            targets = d["targets"].to(self.device)
            
            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)
            loss = outputs[0]
            logits = outputs[1]

            # preds = preds.cpu().detach().numpy()
            _, prediction = torch.max(outputs[1], dim=1)
            targets = targets.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            accuracy = metrics.accuracy_score(targets, prediction)

            acc += accuracy
            losses.append(loss.item())
            
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            counter = counter + 1

            #if idx >= 10:
                #break

        return acc / counter, np.mean(losses)

    def __eval_model(self, data_loader, n_examples):
        model = self.model.eval()
        losses = []
        acc = 0
        counter = 0
    
        with torch.no_grad():
            for idx, d in enumerate(data_loader):
                print('eval_model {}/{}'.format(idx, len(data_loader)))

                if len(d["input_ids"]) < self.BATCH_SIZE:
                    print('Skipped partial batch (got {}, expected {})'.format(len(d["input_ids"]), self.BATCH_SIZE))
                    continue

                input_ids = d["input_ids"].reshape(self.BATCH_SIZE,self.MAX_LEN).to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)
                
                outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)
                loss = outputs[0]
                logits = outputs[1]

                _, prediction = torch.max(outputs[1], dim=1)
                targets = targets.cpu().detach().numpy()
                prediction = prediction.cpu().detach().numpy()
                accuracy = metrics.accuracy_score(targets, prediction)

                acc += accuracy
                losses.append(loss.item())
                counter += 1

                #if idx >= 10:
                    #break

        return acc / counter, np.mean(losses)

    def __get_predictions(self, data_loader):
        model = self.model.eval()
        
        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for idx, d in enumerate(data_loader):

                if len(d["input_ids"]) < self.BATCH_SIZE:
                    print('Skipped partial batch (got {}, expected {})'.format(len(d["input_ids"]), self.BATCH_SIZE))
                    continue

                texts = d["review_text"]
                input_ids = d["input_ids"].reshape(self.BATCH_SIZE,self.MAX_LEN).to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)
                
                outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels = targets)

                loss = outputs[0]
                logits = outputs[1]
                
                _, preds = torch.max(outputs[1], dim=1)

                probs = F.softmax(outputs[1], dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

                #if idx >= 10:
                    #break

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values


