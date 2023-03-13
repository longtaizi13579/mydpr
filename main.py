import random
from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from dpr_model import BiEncoder
from datasets import Dataset
import json
import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--ctx_model_path",
        default='bert-base-uncased',
        type=str,
        help='Model path for context encoder model'
    )
    parser.add_argument(
        "--qry_model_path",
        default='bert-base-uncased',
        type=str,
        help='Model path for qry encoder model'
    )
    parser.add_argument(
        "--path_to_dataset",
        default='./data/',
        type=str,
        help='The path of dataset'
    )
    parser.add_argument(
        "--path_save_model",
        default='./model/',
        type=str,
        help='The path for saving finetuning model'
    )
    parser.add_argument(
        "--valid",
        default=False,
        type=bool,
        help='Exist validation set or not'
    )
    parser.add_argument(
        "--test",
        default=False,
        type=bool,
        help='Exist test set or not'
    )
    parser.add_argument(
        "--encoder_gpu_train_limit",
        default=16,
        type=int,
        help='The number of samples that the gpu can put each time'
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help='Batchsize for training and evaluation'
    )
    parser.add_argument(
        "--lr",
        default=2e-5,
        type=float,
        help='Learning rate for training'
    )
    parser.add_argument(
        "--epoch",
        default=10,
        type=int,
        help='Training epoch number'
    )
    parser.add_argument(
        "--has_cuda",
        default=False,
        type=bool,
        help='Has cuda or not'
    )
    args = parser.parse_args()

    return args


def load_data(args):
    file_addr = args.path_to_dataset
    domains = ['train']
    if args.valid:
        domains.append('valid')
    if args.test:
        domains.append('test')
    all_files = os.listdir(file_addr)
    concrete_dataset = []
    concrete_addr = []
    for every_split_domain in domains:
        for every_file in all_files:
            if every_split_domain in every_file:
                concrete_addr.append(file_addr + '/' + every_file)
                break
    for every_file_addr in concrete_addr:
        file_in = open(every_file_addr, 'r')
        dialogs = json.load(file_in)
        all_query = []
        all_negatives = []
        all_answer = []
        for every_dialog in dialogs:
            if len(every_dialog['hard_negative_ctxs']):
                now_negative = random.choice(every_dialog['hard_negative_ctxs'][:15])
            else:
                now_negative = random.choice(every_dialog['negative_ctxs'][:15])
            all_negatives.append(now_negative['text'].strip())
            all_answer.append(every_dialog['positive_ctxs'][0])
            all_query.append(every_dialog['question'])
        build_dataset = {
            'query': all_query,
            'negative': all_negatives,
            'answer': all_answer,
        }
        concrete_dataset.append(build_dataset)
    return concrete_dataset


def main_work():
    args = get_arguments()
    if args.has_cuda:
        device = 'cpu'
    else:
        device = 'cuda:0'
    data = load_data(args)
    build_train_dataset = data[0]
    train_dataset = Dataset.from_dict(build_train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.valid:
        build_dev_dataset = data[1]
        dev_dataset = Dataset.from_dict(build_dev_dataset)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
        dev_batch_size = args.batch_size
    model = BiEncoder(args)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_epochs = args.epoch
    num_training_steps = num_epochs * len(build_train_dataset['query']) // dev_batch_size
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps,
        num_training_steps=num_training_steps
    )
    save_base_directory = args.path_save_model
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.ctx_model_path)
    qry_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.qry_model_path)
    for epoch in range(num_epochs):
        whole_loss = 0
        whole_num = 0
        model.train()
        for index, sample in enumerate(train_dataloader):
            positive = torch.tensor(list(range(len(sample['answer'])))).to(device)
            sample['answer'].extend(sample['negative'])
            ctx = ctx_tokenizer(sample['answer'], padding=True, truncation=True, max_length=512,
                                return_tensors='pt')  # max 512
            qry = qry_tokenizer(sample['query'], padding=True, truncation=True, max_length=512, return_tensors='pt')
            ctx = {k: v.to(device) for k, v in ctx.items()}
            qry = {k: v.to(device) for k, v in qry.items()}
            batch = {'ctx': ctx, 'qry': qry, 'positive': positive}
            loss, accuracy = model(batch)
            loss.backward()
            whole_num += 1
            whole_loss += loss
            if (index - 1) % 10 == 0:
                print(f'{whole_num} loss: ', whole_loss / whole_num)
                print(f'lr: {lr_scheduler.get_last_lr()}')
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        pt_save_directory = save_base_directory + str(epoch) + f'_{str(round(whole_accuracy, 4))}'
        qry_tokenizer.save_pretrained(pt_save_directory + '/qry')
        ctx_tokenizer.save_pretrained(pt_save_directory + '/ctx')
        model.save_pretrained(pt_save_directory)
        data = load_data(args)
        build_train_dataset = data[0]
        train_dataset = Dataset.from_dict(build_train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        if args.valid:
            model.eval()
            correct_num = 0
            whole_num = 0
            whole_loss = 0
            for index, sample in enumerate(dev_dataloader):
                with torch.no_grad():
                    positive = torch.tensor(list(range(len(sample['answer'])))).to(device)
                    sample['answer'].extend(sample['negative'])
                    ctx = ctx_tokenizer(sample['answer'], padding=True, truncation=True, max_length=512,
                                        return_tensors='pt')
                    qry = qry_tokenizer(sample['query'], padding=True, truncation=True, max_length=512,
                                        return_tensors='pt')
                    ctx = {k: v.to(device) for k, v in ctx.items()}
                    qry = {k: v.to(device) for k, v in qry.items()}
                    batch = {'ctx': ctx, 'qry': qry, 'positive': positive}
                    loss, accuracy = model(batch)
                    whole_loss += loss
                    whole_num += dev_batch_size
                    correct_num += dev_batch_size * accuracy
            whole_accuracy = float(correct_num) / whole_num
            print('eval loss: ', whole_loss)
            print('eval accuracy: ', whole_accuracy)
            build_dev_dataset = data[1]
            dev_dataset = Dataset.from_dict(build_dev_dataset)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)


if __name__ == "__main__":
    main_work()
