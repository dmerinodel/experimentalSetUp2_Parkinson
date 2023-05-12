import pandas as pd
import os
import torch
from torch import nn, optim
import time
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

from parkinsonClassification.flow import train
from parkinsonClassification.dataset import SoundDS
from parkinsonClassification.models import Model1, count_parameters
from torch.utils.data import DataLoader
import parkinsonClassification.dataug as aug

# Antes de nada nos vamos a quedar con los nombres de los pacientes que van en
# train-test-val

# Leemos y guardamos PCGita
excel_path = 'datasets/PCG_Parkinson_ds/PCGITA_metadata.xlsx'
names = pd.read_excel(excel_path)
names = names[['RECORDING ORIGINAL NAME', 'SEX']]
names['classID'] = [0 if 'C' in name else 1 for name in names['RECORDING ORIGINAL NAME']]

# Leemos y guardamos UEX
excel_path1 = 'datasets/UEX_Parkinson_ds/UEX_metadata.xlsx'
names1 = pd.read_excel(excel_path1)
names1 = names1[['Identificador', 'Genero']]
newgen = ['M' if gen == 0 else 'F' for gen in names1['Genero']] # Unificamos notación de género
names1['Genero'] = newgen
names1['classID'] = [0 if 'C' in name else 1 for name in names1['Identificador']]
names1 = names1.rename(columns={'Identificador': 'RECORDING ORIGINAL NAME', 'Genero': 'SEX'})

# Muestreamos 90-10
train_names = names.sample(frac=0.9)
train_names1 = names1.sample(frac=0.9)
val_names = pd.concat([names, train_names], ignore_index=True).drop_duplicates(keep=False)
val_names1 = pd.concat([names1, train_names1], ignore_index=True).drop_duplicates(keep=False)

# Comprobaciones
tms = len(train_names[(train_names['SEX'] == 'M') & (train_names['classID'] == 0)])
tme = len(train_names[(train_names['SEX'] == 'M') & (train_names['classID'] == 1)])
tfs = len(train_names[(train_names['SEX'] == 'F') & (train_names['classID'] == 0)])
tfe = len(train_names[(train_names['SEX'] == 'F') & (train_names['classID'] == 1)])

vms = len(val_names[(val_names['SEX'] == 'M') & (val_names['classID'] == 0)])
vme = len(val_names[(val_names['SEX'] == 'M') & (val_names['classID'] == 1)])
vfs = len(val_names[(val_names['SEX'] == 'F') & (val_names['classID'] == 0)])
vfe = len(val_names[(val_names['SEX'] == 'F') & (val_names['classID'] == 1)])

print('Train: ', len(train_names))
print(f'Hombres sanos: {tms} - Hombres enfermos {tme}')
print(f'Mujeres sanas: {tfs} - Mujeres enfermas {tfe}')
print('Val: ', len(val_names))
print(f'Hombres sanos: {vms} - Hombres enfermos {vme}')
print(f'Mujeres sanas: {vfs} - Mujeres enfermas {vfe}')

tms1 = len(train_names1[(train_names1['SEX'] == 'M') & (train_names1['classID'] == 0)])
tme1 = len(train_names1[(train_names1['SEX'] == 'M') & (train_names1['classID'] == 1)])
tfs1 = len(train_names1[(train_names1['SEX'] == 'F') & (train_names1['classID'] == 0)])
tfe1 = len(train_names1[(train_names1['SEX'] == 'F') & (train_names1['classID'] == 1)])

vms1 = len(val_names1[(val_names1['SEX'] == 'M') & (val_names1['classID'] == 0)])
vme1 = len(val_names1[(val_names1['SEX'] == 'M') & (val_names1['classID'] == 1)])
vfs1 = len(val_names1[(val_names1['SEX'] == 'F') & (val_names1['classID'] == 0)])
vfe1 = len(val_names1[(val_names1['SEX'] == 'F') & (val_names1['classID'] == 1)])

print('Train: ',len(train_names1))
print(f'Hombres sanos: {tms1} - Hombres enfermos {tme1}')
print(f'Mujeres sanas: {tfs1} - Mujeres enfermas {tfe1}')
print('Val: ', len(val_names1))
print(f'Hombres sanos: {vms1} - Hombres enfermos {vme1}')
print(f'Mujeres sanas: {vfs1} - Mujeres enfermas {vfe1}')

# ----------------------------
# Preparando datos de entrenamiento desde los metadatos
# ----------------------------
download_path = 'datasets/PCG_Parkinson_ds/'

# Leemos archivo de metadatos
metadata_file = os.path.join(download_path, 'metadata/PCGITA_metadata.csv')
df = pd.read_csv(metadata_file)

# Construimos el path de los archivos añadiendo el nombre de las carpetas
df['relative_path'] = '/' + df['fold'].astype(str) + '/' + df['RECORDING_ORIGINAL_NAME'].astype(str) + '.wav'

# Nos quedamos con las columnas que importan
df = df[['relative_path', 'classID', 'sex']]

download_path1 = 'datasets/UEX_Parkinson_ds/'

# Leemos archivo de metadatos
metadata_file1 = os.path.join(download_path1, 'metadata/UEX_metadata.csv')
df1 = pd.read_csv(metadata_file1)

# Construimos el path de los archivos añadiendo el nombre de las carpetas
df1['relative_path'] = '/' + df1['fold'].astype(str) + '/' + df1['RECORDING_ORIGINAL_NAME'].astype(str) + '.wav'

# Nos quedamos con las columnas que importan
df1 = df1[['relative_path', 'classID', 'sex']]

# Ahora vamos a quedarnos con las grabaciones correspondientes a los nombres
# seleccionados en train, val y test.

df_train = pd.DataFrame()
for name in train_names['RECORDING ORIGINAL NAME']:
    df_train = pd.concat([df_train, df[df['relative_path'].str.contains(name)]], ignore_index=True)

df_val = pd.DataFrame()
for name in val_names['RECORDING ORIGINAL NAME']:
    df_val = pd.concat([df_val, df[df['relative_path'].str.contains(name)]], ignore_index=True)

# Comprobaciones.
print('Datos entrenamiento PCGita: ', len(df_train))
print('Sanos PCGita: ', len(df_train[df_train['classID'] == 0]))
print('Enfermos PCGita: ', len(df_train[df_train['classID'] == 1]))
print('-------------------------------------------')
print('Datos validación PCGita: ', len(df_val))
print('Sanos PCGita: ', len(df_val[df_val['classID'] == 0]))
print('Enfermos PCGita: ', len(df_val[df_val['classID'] == 1]))

df_train1 = pd.DataFrame()
for name in train_names1['RECORDING ORIGINAL NAME']:
    df_train1 = pd.concat([df_train1, df1[df1['relative_path'].str.contains(name)]], ignore_index = True)

df_val1 = pd.DataFrame()
for name in val_names1['RECORDING ORIGINAL NAME']:
    df_val1 = pd.concat([df_val1, df1[df1['relative_path'].str.contains(name)]], ignore_index = True)

print('Datos entrenamiento UEx: ', len(df_train1))
print('Sanos UEx: ', len(df_train1[df_train1['classID'] == 0]))
print('Enfermos UEx: ', len(df_train1[df_train1['classID'] == 1]))
print('-------------------------------------------')
print('Datos validación UEx: ', len(df_val1))
print('Sanos UEx: ', len(df_val1[df_val1['classID'] == 0]))
print('Enfermos UEx: ', len(df_val1[df_val1['classID'] == 1]))


data_path = 'datasets/PCG_Parkinson_ds/A'
data_path1 = 'datasets/UEX_Parkinson_ds/A'

train_ds = SoundDS(df_train, data_path)
val_ds = SoundDS(df_val, data_path)
train_ds1 = SoundDS(df_train1, data_path1)
val_ds1 = SoundDS(df_val1, data_path1)

# Juntamos ambos datasets
merged_train = [train_ds, train_ds1]
train_m = ConcatDataset(merged_train)
merged_val = [val_ds, val_ds1]
val_m = ConcatDataset(merged_val)

print('INPUT: ', train_ds.__getitem__(0)[0].shape)

# Cargamos los datos
bs = 64
train_dl = DataLoader(train_m, batch_size=bs, shuffle=True)
val_dl = DataLoader(val_m, batch_size=bs, shuffle=True)
dataloaders = {'train': train_dl, 'valid': val_dl}

# Modelo
myModel = Model1()
# Transfer learning TODO: definir con flag y pasar por parámetro
weights = torch.load('models/pretrain.pt')
myModel.load_state_dict(weights)
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
train_time = (final-inicio)/60
print(f'Ha tardado {train_time} minutos.')

fig = plt.figure(figsize=(16, 9))
plt.plot(loss_dict["valid"])
plt.plot(loss_dict["train"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["valid loss", "training loss"])
plt.title('Model1')
plt.savefig("figures/pcg_loss.png")
plt.close()

fig = plt.figure(figsize=(16, 9))
plt.plot(loss_dict["valid_acc"])
plt.plot(loss_dict["train_acc"])
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend(["valid accuracy", "train accuracy"])
plt.title('Model1')
plt.savefig("figures/pcg_acc.png")
plt.close()

res = open('results/pcg.txt', 'a')
res.write(f'Tiempo Model1: {train_time} minutos\n')
res.write('train_loss \t valid_loss \t train_acc \t valid_acc\n')
for dat1, dat2, dat3, dat4 in zip(loss_dict['train'], loss_dict['valid'],
                                  loss_dict['train_acc'], loss_dict['valid_acc']):
    res.write(f'{dat1} \t {dat2} \t {dat3} \t {dat4}\n')
res.close()

