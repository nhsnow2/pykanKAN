# --- imports -----------------------------------------------------------------------------------------------------------------
# Models
import torch
import kan
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ConstantKernel, CompoundKernel, WhiteKernel
# Data handling
import numpy as np
import pandas as pd
import time as timer

start_time = timer.time()

# --- Load Data ---------------------------------------------------------------------------------------------------------------
input_data = []
for i in range(1,8):
    input_data.append(np.load(f"../samples/SampleXS_{i}.npy"))
input_data = np.array(input_data)
output_data = np.load(f"../samples/SampleOutput.npy")

# --- Flatten Data
#print(f"Pre-Inputs Shape:\t{[i.shape for i in input_data]}")

print(f"Recieved Training Data:\t(t={timer.time()-start_time})")

print("Inputs")
print("_____________________________")
print(f"Pre:\t{input_data.shape}") #type:ignore
flattened_inputs = input_data.reshape(500, 81*16*7) #type:ignore
print(f"Post:\t{flattened_inputs.shape}") #type:ignore

print("_____________________________")
print("Outputs")
print("_____________________________")
print(f"Pre:\t{[output_data.shape]}")
flattened_outputs = output_data.reshape((500,81*16))
print(f"Post:\t{flattened_outputs.shape}")

# --- Sample Splitting
print("Splitting into Train and Test sets:")
train_X, test_X = train_test_split(flattened_inputs, test_size=0.2)
train_Y, test_Y = train_test_split(flattened_outputs, test_size=0.2)

print(f"Train X:\t{train_X.shape}\tTest X:\t{test_X.shape}")
print(f"Train Y:\t{train_Y.shape}\tTest Y:\t{test_Y.shape}")

# --- Data Scaling
print("Performing standard scaling:")
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
#
train_XS = X_scaler.fit_transform(train_X)
train_YS = Y_scaler.fit_transform(train_Y)

test_XS = X_scaler.transform(test_X)
test_YS = Y_scaler.transform(test_Y)

# --- Conversion to Torch tensors
print("Converting to Torch tensors:")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device:{device}")

train_XST = torch.tensor(train_XS, dtype=torch.float32, device=device)
train_YST = torch.tensor(train_YS, dtype=torch.float32, device=device)
test_XST = torch.tensor(test_XS, dtype=torch.float32, device=device)
test_YST = torch.tensor(test_YS, dtype=torch.float32, device=device)

DATASET = {'train_input':train_XST, 'test_input':train_YST, 'train_label':test_XST, 'test_label':test_YST}

# --- KAN Model ---------------------------------------------------------------------------------------------------------------
print(f"Initializing Model:\t(t={timer.time()-start_time})")
model = kan.KAN([train_XST.shape[1], train_XST.shape[1], train_YST.shape[1]], grid=2, k=3, auto_save=True, device=device)
print(f"Incorporating data size:\t(t={timer.time()-start_time})")
model(DATASET['train_input'])
print(f"Training Model...")
model.fit(DATASET, opt='LBFGS', steps=75, lamb=2.043e-05, lamb_entropy=5.03464, lr=1.5, reg_metric='edge_forward_sum')
print("Training Complete.\t(t={timer.time()-start_time})")

print("Model now saved under {./models}.")