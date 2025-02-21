import os
import argparse
import numpy as np
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from model import MyModel
import dataset


def train(args, model, train_dataset, validation_dataset, binary_classification=False):
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr= args.learning_rate)
    
    criterion = torch.nn.MSELoss()
    binary_criterion = torch.nn.BCELoss()

    train_dataLoader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=1)
    
    validation_dataLoader = torch.utils.data.DataLoader(validation_dataset,
                                                        batch_size=args.validation_batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=1)
    
    start_epoch = 0
    
    # Resume training from checkpoint
    if args.resume_path is not None:
        print("Resuming training from checkpoint in ", args.resume_path)
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            checkpoint = torch.load(args.resume_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        
    
    start_time = time.time()    
    for epoch in range(start_epoch, args.epochs + start_epoch):
        
        # Training
        # ------------------------
        model.train()
        train_loss = 0
        for image,label in train_dataLoader:
            
            if torch.cuda.is_available():
                image   = image.cuda()
                label = label.cuda()
            
            output = model(image)
            if binary_classification:
                loss = binary_criterion(output.flatten(), label.float())
            else:
                loss = criterion(output.flatten(), label.float())


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss
            
        train_loss = train_loss/len(train_dataset)
        
        # Validation 
        # ------------------------
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for image,label in validation_dataLoader:
            
                if torch.cuda.is_available():
                    image   = image.cuda()
                    label = label.cuda()
                
                output = model(image).flatten()
                if binary_classification:
                    loss = binary_criterion(output, label.float())
                else:
                    loss = criterion(torch.Tensor(np.array([0.0 if value <0.5 else 1.0 for value in output])), label.float())
                optimizer.zero_grad()

                validation_loss += loss
                
            validation_loss = validation_loss/len(validation_dataset)
        
        # log to wandb every 10th epoch
        if (epoch !=0 and epoch % 1 == 0) or (epoch == args.epochs + start_epoch - 1):
            wandb.log({
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "time since starting": time.time() - start_time,
                }
            )
            print("***************************************")
            print("epoch: ", epoch)
            print("training Loss: ", train_loss.item())
            print("Validation Loss: ", validation_loss.item())
        # Print current loss and save dict every 10th epoch
        # ------------------------
        if (epoch != 0 and epoch % 10 == 0) or (epoch == args.epochs + start_epoch - 1):
            print("epoch: ", epoch)
            print("training Loss: ", train_loss.item())
            print("Validation Loss: ", validation_loss.item())
            
            state = {
                'epoch'    : epoch,
                'train_loss': train_loss,
                'validation_loss': validation_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            
            torch.save(state, os.path.join(args.model_path, 'model-{}.pt'.format(state['epoch'])))

def wandb_init(args):
    if not os.path.exists('./run_number.txt'):
        f = open("run_number.txt", "w")
        f.write("0")
        f.close()
    with open("run_number.txt", "r+") as f:
        runs = f.read()
        run = int(runs.split(',')[-1]) + 1
        f.write(',' + str(run))
        wandb.init(
            # set the wandb project where this run will be logged
            project="cat_or_dog",

            # track hyperparameters and run metadata
            config={
                "name": "CNN-7-5-1" + str(run),
                "learning_rate": args.learning_rate,
                "architecture": "CNN-7-5-1",
                "dataset": "cats_and_dogs_binary",
                "epochs": args.epochs,
            }
        )

def main(args):
    torch.cuda.empty_cache()
    wandb_init(args)

    train_set       = dataset.MyDataset(dataset.CAT_TRAIN_PATH, dataset.DOG_TRAIN_PATH)
    validation_set  = dataset.MyDataset(dataset.CAT_VAL_PATH, dataset.DOG_VAL_PATH)
    model           = MyModel(color=True, bias=True)

    train(args, model, train_set, validation_set, binary_classification=args.binary_classification)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train_dir'    ,help='train data directory'                           ,dest='train_path'            ,type=str   ,default='./train/')
    # parser.add_argument('--val_dir'      ,help='validation data directory'                      ,dest='validation_path'       ,type=str   ,default='./validation/')
    parser.add_argument('--model_dir'    ,help='model directory'                                ,dest='model_path'            ,type=str   ,default='./models/')
    parser.add_argument('--resume_dir'   ,help='if resuming, set path to checkpoint'            ,dest='resume_path'           ,type=str   ,default=None)
    parser.add_argument('--train_batch'  ,help='train batch size'                               ,dest='train_batch_size'      ,type=int   ,default=350)
    parser.add_argument('--val_batch'    ,help='validation batch size'                          ,dest='validation_batch_size' ,type=int   ,default=50)
    parser.add_argument('--epochs'       ,help='number of epochs to train for'                  ,dest='epochs'                ,type=int   ,default=10001)
    parser.add_argument('--learning_rate',help='learning rate'                                  ,dest='learning_rate'         ,type=float ,default=1e-3)
    parser.add_argument('--binary'       ,help='is this a binary classification?'               ,dest='binary_classification' ,type=bool  ,default=True)
    args = parser.parse_args()
    
    main(args)