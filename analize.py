import numpy as np
import csv

from test import main_test


csv_file = True
def main_analize():
    correct = 0
    mistake = 0
    if not csv_file:
        results= np.load('analysis.npy')
        labels = np.array([])
        outputs = np.array([])
        labels = results[0]
        outputs = results[1]
        for idx, label in enumerate(labels):
            output = outputs[idx]
            
            if label == output:
                correct += 1
            else:
                mistake += 1

    else:
        with open('analysis.csv', newline='') as file:
            labels = []
            outputs = []
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                labels.append(row[0])
                outputs.append(row[1])
            for idx, label in enumerate(labels):
                output = outputs[idx]
                
                if label == output:
                    correct += 1
                else:
                    mistake += 1
    return correct, mistake

if __name__ == "__main__":
    most_correct = 0
    best_iteration = 0
    for i in range(10, 500, 10):
        main_test(debug=False, model_number=i)
        correct, mistake =main_analize()
        if correct > most_correct:
            most_correct = correct
            best_iteration = i
            print("----- new best iteration: ", best_iteration " -----")
            print("correct: ", correct)

            print("mistake: ", mistake)
            print("accuracy: ", 100*correct/(correct+mistake), "%")