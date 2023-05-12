import numpy as np
import time
import os
import torch
import pickle

def train(n_epochs,
          dataloaders,
          model,
          criterion,
          optimizer,
          use_cuda):

    if use_cuda:
        model = model.cuda()

    loss_dict = {'train': [], 'valid': [], 'valid_acc': [], 'train_acc': []}
    valid_loss_min = np.Inf
    prev_save = ""
    print("criterion: {}".format(criterion))

    for e in range(1, n_epochs + 1):
        start = time.time()
        train_loss, valid_loss, n_corr, n_t_corr = 0., 0., 0, 0

        # ---------------  TRAIN THE MODEL  ---------------
        model.train()
        for data, target in dataloaders['train']:
            if use_cuda:
                data = data.cuda()
                target = target.cuda()  # shape: [batch_size]

            optimizer.zero_grad()
            output = model(data)  # shape: [batch_size, n_classes]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            output = output.cpu().detach().numpy()
            n_t_corr += int(sum([np.argmax(pred) == target[i] for i, pred in enumerate(output)]))

        # ---------------  VALIDATE THE MODEL  ---------------
        model.eval()
        for data, target in dataloaders['valid']:
            if use_cuda:
                data = data.cuda()
                target = target.cuda()  # shape: [batch_size]

            output = model(data)  # [batch_size, n_classes]
            loss = criterion(output, target)
            valid_loss += loss.item()
            output = output.cpu().detach().numpy()
            n_corr += int(sum([np.argmax(pred) == target[i] for i, pred in enumerate(output)]))

        train_loss = train_loss / len(dataloaders['train'].dataset)
        train_acc = n_t_corr / len(dataloaders['train'].dataset)
        valid_loss = valid_loss / len(dataloaders['valid'].dataset)
        valid_acc = n_corr / len(dataloaders['valid'].dataset)

        loss_dict['train'].append(train_loss)
        loss_dict['train_acc'].append(train_acc)
        loss_dict['valid'].append(valid_loss)
        loss_dict['valid_acc'].append(valid_acc)

        # Log result each epoch
        print('Epoch: %d/%d\t Train Loss: %.5f\t Valid Loss: %.5f\t Valid Acc: %.4f\t elapsed time: %.1fs' % (
                e, n_epochs, train_loss, valid_loss, valid_acc, time.time() - start))
        # TODO: Implementar métricas con torch.metrics
        # Save model if the current validation loss is lower than the previous validation loss
        if valid_loss < valid_loss_min:
            if prev_save:
                os.remove("models/model" + prev_save + ".pt")
                os.remove("models/loss_dict" + prev_save + ".pkl")
            prev_save = "_" + str(e)
            torch.save(model.state_dict(), "models/model" + prev_save + ".pt")
            pickle.dump(loss_dict, open("models/loss_dict" + prev_save + ".pkl", "wb"))
            valid_loss_min = valid_loss

    return loss_dict, model
