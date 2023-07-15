from datasetSQUAD import SquadDataset
from utils import process_train, process_val

import argparse

from transformers import  RobertaForQuestionAnswering, BertForQuestionAnswering, ElectraForQuestionAnswering
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser('Question answering in Squad 2.0')
    parser.add_argument('--model', type=str, default = 'bert-base-uncased', help = 'name of pre-trained model')
    parser.add_argument('--batch_size', type=int, default = 16, help = 'The number of sample per batch among all devices')
    parser.add_argument('--learning_rate', type=float, default = 2e-5)
    parser.add_argument('--num_epochs', type=int, default = 4)
    parser.add_argument('--saved_path', type=str, default = 'logs')
    args = parser.parse_args()
    return args



def train(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available()else 'cpu')

    if opt.model == "roberta-base":
        model = RobertaForQuestionAnswering.from_pretrained(opt.model).to(device)
    elif opt.model == "google/electra-base-discriminator":
        model = ElectraForQuestionAnswering.from_pretrained(opt.model).to(device)
    else:
        model = BertForQuestionAnswering.from_pretrained(opt.model).to(device)


    optim = AdamW(model.parameters(), lr = opt.learning_rate)

    train_texts, train_queries, train_answers = process_train()
    val_texts, val_queries, val_answers = process_val()

    train_dataset = SquadDataset(train_texts, train_queries, train_answers, model_pretrained = opt.model)
    val_dataset = SquadDataset(val_texts, val_queries, val_answers, model_pretrained = opt.model)


    train_loader = DataLoader(train_dataset, batch_size = opt.batch_size, shuffle = True)
    
    val_loader = DataLoader(val_dataset, batch_size = opt.batch_size, shuffle = False)

    epochs = opt.num_epochs

    # Initialize train and validation losses lists
    train_losses = []
    val_losses = []

    print_every = 1000
    for epoch in range(epochs):
        # Set model in train mode
        model.train()
        
        loss_of_epoch = 0
        
        print("----------Train----------")
        for batch_idx,batch in enumerate(tqdm(train_loader)):

            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]

            loss.backward()

            # update weights
            optim.step()
            # total loss
            loss_of_epoch += loss.item()

            if (batch_idx+1) % print_every == 0:
                print("Batch {:} / {:}".format(batch_idx+1,len(train_loader)),"\nLoss:", loss.item(),"\n")

        loss_of_epoch /= len(train_loader)
        train_losses.append(loss_of_epoch)

        # Set model in evaluation mode
        model.eval() 

        print("----------Evaluate----------")
        loss_of_epoch = 0

        for batch_idx, batch in enumerate(tqdm(val_loader)): 
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
      
                outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                loss = outputs[0]

                # Find the total loss
                loss_of_epoch += loss.item()

            if (batch_idx+1) % print_every == 0:
                print("Batch {:} / {:}".format(batch_idx+1, len(val_loader)),"\nLoss:", loss.item(),"\n")
          # Print each epoch's time and train/val loss 
        loss_of_epoch /= len(val_loader)
        val_losses.append(loss_of_epoch)
        
        print("------- Epoch ", epoch+1 ," -------"", Training Loss:", train_losses[-1], ", Validation Loss:", val_losses[-1], "\n")

        saved_path = opt.saved_path + f'/{epoch}.pt'

        torch.save({
            'model_state_dict': model.state_dict(),
            'learning rate': opt.learning_rate,
            'batch_size': opt.batch_size,
            'loss_train': train_losses[-1],
            'loss_val':val_losses[-1]
            }, saved_path)
if __name__ == '__main__':
    opt = get_args()
    print('\n')
    print("Describe model: ",  opt)
    print("\n")
    train(opt)