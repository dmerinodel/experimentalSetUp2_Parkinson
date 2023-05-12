import pandas as pd
import os
import torch
from torch import nn, optim
import time
import matplotlib.pyplot as plt

from parkinsonClassification.flow import train
from parkinsonClassification.dataset import SoundDS
from parkinsonClassification.models import Model1, count_parameters
from torch.utils.data import DataLoader

# Lectura de excel
excel_path = 'datasets/PreTrain_VoicePat_ds/PreTrain_metadata.xlsx'
names = pd.read_excel(excel_path)
names = names[['Identificador']]
names['classID'] = [0 if 'C' in name else 1 for name in names['Identificador']]
print(f'Total de pacientes: {len(names)}')
print(names.head())

# Elegimos pacientes para train-val, 90-10.
train_names = names.sample(frac=0.9)
names = pd.concat([names, train_names], ignore_index=True).drop_duplicates(keep=False)
val_names = names
print('Pacientes train: ', len(train_names))
print('Pacientes val: ', len(val_names))
print(train_names.head())

# Preparando desde los metadatos
download_path = 'datasets/PreTrain_VoicePat_ds/'
metadata_file = os.path.join(download_path, 'metadata/PreTrain_metadata.csv')
df = pd.read_csv(metadata_file)
# Construimos el path de los archivos añadiendo el nombre de las carpetas
df['relative_path'] = '/' + df['fold'].astype(str) + '/' + df['RECORDING_ORIGINAL_NAME'].astype(str) + '.wav'
# Nos quedamos con las columnas que importan
df = df[['relative_path', 'classID']]
print(df.head())

# Seleccionamos grabaciones correspondientes a los pacientes de train y val
df_train = pd.DataFrame()
for name in train_names['Identificador']:
    df_train = pd.concat([df_train, df[df['relative_path'].str.contains(name)]], ignore_index=True)

df_val = pd.DataFrame()
for name in val_names['Identificador']:
    df_val = pd.concat([df_val, df[df['relative_path'].str.contains(name)]], ignore_index=True)

# Comprobaciones.
print('Datos entrenamiento: ', len(df_train))
print('Sanos: ', len(df_train[df_train['classID'] == 0]))
print('Enfermos: ', len(df_train[df_train['classID'] == 1]))
print('-------------------------------------------')
print('Datos validación: ', len(df_val))
print('Sanos: ', len(df_val[df_val['classID'] == 0]))
print('Enfermos: ', len(df_val[df_val['classID'] == 1]))

data_path = 'datasets/PreTrain_VoicePat_ds/A'

train_ds = SoundDS(df_train, data_path)
val_ds = SoundDS(df_val, data_path)

print('INPUT: ', train_ds.__getitem__(0)[0].shape)

# Cargamos los datos
bs = 64
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=bs, shuffle=True)
dataloaders = {'train': train_dl, 'valid': val_dl}

# Modelo
myModel = Model1()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
print(next(myModel.parameters()).device)
print(f'El modelo tiene {count_parameters(myModel)} parámetros')

# Hiperparámetros
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(myModel.parameters(), lr=2e-5)

# Entrenamiento
n_epochs = 100
use_cuda = torch.cuda.is_available()
print('GPU:', use_cuda)
inicio = time.time()
loss_dict, model = train(n_epochs=n_epochs,
                         dataloaders=dataloaders,
                         model=myModel,
                         criterion=criterion,
                         use_cuda=use_cuda,
                         optimizer=optimizer)
final = time.time()
train_time = (final - inicio)/60
print(f'Ha tardado {train_time} minutos.')

# Exportamos pesos para el resto de experimentos
save_path = 'models/pretrain.pt'

fig = plt.figure(figsize=(16, 9))
plt.plot(loss_dict["valid"])
plt.plot(loss_dict["train"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["valid loss", "training loss"])
plt.title('Model1')
plt.savefig("figures/pretrain_loss.png")
plt.close()

fig = plt.figure(figsize=(16, 9))
plt.plot(loss_dict["valid_acc"])
plt.plot(loss_dict["train_acc"])
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend(["valid accuracy", "train accuracy"])
plt.title('Model1')
plt.savefig("figures/pretrain_acc.png")
plt.close()

res = open('results/pretrain.txt', 'a')
res.write(f'Tiempo Model1: {train_time} minutos\n')
res.write('train_loss \t valid_loss \t train_acc \t valid_acc\n')
for dat1, dat2, dat3, dat4 in zip(loss_dict['train'], loss_dict['valid'],
                                  loss_dict['train_acc'], loss_dict['valid_acc']):
    res.write(f'{dat1} \t {dat2} \t {dat3} \t {dat4}\n')
res.close()
