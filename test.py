
import os, cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("model2.h5")

test_dataset_dir = "./test_dataset/images"
class_mapping = {'BareLand': 0, 'Commercial': 1, 'DenseResidential': 2, 'Desert': 3, 'Farmland': 4, 'Forest': 5, 'Industrial': 6, 'Meadow': 7, 'MediumResidential': 8, 'Mountain': 9, 'River': 10, 'SparseResidential': 11}
class_mapping_inv = dict(zip(class_mapping.values(), class_mapping.keys()))
X = []
y = []
index = 0
for class_name in os.listdir(test_dataset_dir):
    
    y_val = [0.0 for i in range(len(class_mapping))]
    y_val[index] = 1.0
    index += 1

    print(class_name)

    for image in os.listdir(test_dataset_dir + "/" + class_name):
        X.append(cv2.resize(cv2.imread(test_dataset_dir + "/" + class_name + "/" + image), (225, 225)))
        y.append(y_val)

X = np.array(X, dtype=np.uint8)
#y = np.array(y, dtype=float)

y_pred = model.predict(X)
correct = 0
for i in range(len(y)):
    if list(y_pred[i]).index(max(y_pred[i])) ==  y[i].index(max(y[i])):
        correct += 1
        print("Correct -", max(y_pred[i]),
        class_mapping_inv[ y[i].index(max(y[i]))])
    elif ("Resident" in class_mapping_inv[ y[i].index(max(y[i]))] and "Resident" in class_mapping_inv[ list(y_pred[i]).index(max(y_pred[i])) ])\
    or ("Industr" in class_mapping_inv[ y[i].index(max(y[i]))] and "Resident" in class_mapping_inv[ list(y_pred[i]).index(max(y_pred[i])) ])\
    or ("Resident" in class_mapping_inv[ y[i].index(max(y[i]))] and "Industr" in class_mapping_inv[ list(y_pred[i]).index(max(y_pred[i])) ]):
        correct += 1
        print("Correct (Common Residential/Industrial)-", max(y_pred[i]) )
    else:
        print("Incorrect -", max(y_pred[i]),
        "True -", class_mapping_inv[ y[i].index(max(y[i]))], "Pred -", class_mapping_inv[ list(y_pred[i]).index(max(y_pred[i])) ])

print("Accuracy -", correct/len(y))
print("Correct -", correct, "out of", len(y))