import os
import argparse
import numpy as np
import glob
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import csv

from model import MyModel
import dataset

def main_test(debug=False, model_number=400):

    model = MyModel()
    test_dataset    = dataset.MyDataset(dataset.CAT_TEST_PATH, dataset.DOG_TEST_PATH)
    if debug:
        test_dataLoader = torch.utils.data.DataLoader(  test_dataset,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=0)
    else:
        test_dataLoader = torch.utils.data.DataLoader(  test_dataset,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=0)

    if torch.cuda.is_available():
        model = model.cuda()

    checkpoint =    torch.load("models/model-%s.pt" % model_number)
    model.load_state_dict(checkpoint['state_dict'])

    # test
    # ------------------------
    model.eval()
    output_list = np.array([])
    label_list = np.array([])
    with open('analysis.csv', 'w', newline='') as f:
        writter = csv.writer(f, delimiter=',')
        for image,label in test_dataLoader:
            
            if torch.cuda.is_available():
                image   = image.cuda()
                label = label.cuda()
            output = model(image).flatten()
            label_list = np.concatenate([label_list, label.cpu().numpy()])
            output_list = np.concatenate([output_list, output.detach().cpu().numpy()])
            if debug:
                print("-------------------------------------------------")
                print("label: ", "cat " if label == 0 else "dog")
                print("output: ", "cat " if output == 0 else "dog")
                cv2.imshow("image", image.cpu().numpy().squeeze().transpose(1,2,0))
                cv2.waitKey(0)
            writter.writerow([label.cpu().numpy()[0], 0 if output.detach().cpu().numpy()<0.5 else 1])
        # final_list = np.array([label_list, output_list])
        # np.save(f, final_list)
            
        
            

if __name__ == "__main__":
    main_test(debug=False, model_number=400)
    