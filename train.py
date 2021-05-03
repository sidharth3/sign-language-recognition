import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import logging
import os
import torch.optim as optim
from torch.utils.data import Dataset
import utils
from configs import Config
from model import GCNMultiBlock
from dataloader import Sign_Dataset
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run(file_split, pose_directory, configs, save_model=None):
    
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages

    # setup dataset
    
    train_dataset = Sign_Dataset(file_name_index=file_split, split=['train', 'val'], pose_directory=pose_directory, img_transforms=None, video_transforms=None, num_samples=num_samples )

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                                    shuffle=True)

    val_dataset = Sign_Dataset(file_name_index=file_split, split='test', pose_directory=pose_directory,
                               img_transforms=None, video_transforms=None,
                               num_samples=num_samples, sampling_method='k_copies')
    
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                                  shuffle=True)

    # setup the model
    model = GCNMultiBlock(input_feature=num_samples*2, hidden_feature=num_samples*2,
                         num_class=len(train_dataset.label_encoder.classes_), p_dropout=drop_p, num_stage=num_stages).cuda()

    # setup training parameters, learning rate, optimizer, scheduler
    lr = configs.init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=configs.adam_eps, weight_decay=configs.adam_weight_decay)
    
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_val_losses = []
    epoch_val_scores = []

    best_test_acc = 0
    # start training
    current_epoch = 1
    for epoch in range(int(epochs)):
        # train, test model

        print('Training now...Epoch[{}/{}]'.format(current_epoch, epochs))
        train_losses, train_scores, train_gts, train_preds = train(log_interval, model,
                                                                   train_data_loader, optimizer, epoch)
        print('Testing now...Epoch[{}/{}]'.format(current_epoch, epochs))
        val_loss, val_score, val_gts, val_preds, incorrect_samples = validation(model,
                                                                                val_data_loader, epoch,
                                                                                save_to=save_model)
        
        current_epoch+=1
        #append training-testing loss/accuracy data
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(train_scores)
        epoch_val_losses.append(val_loss)
        epoch_val_scores.append(val_score[0])
        
        #save the data for plotting later
        np.save('output/epoch_training_losses.npy', np.array(epoch_train_losses))
        np.save('output/epoch_training_scores.npy', np.array(epoch_train_scores))
        np.save('output/epoch_test_loss.npy', np.array(epoch_val_losses))
        np.save('output/epoch_test_score.npy', np.array(epoch_val_scores))

        if val_score[0] > best_test_acc:
            best_test_acc = val_score[0]
            best_epoch_num = epoch

            torch.save(model.state_dict(), os.path.join('checkpoints', subset, 'gcn_epoch={}_val_acc={}.pth'.format(
                best_epoch_num, best_test_acc)))

    utils.plot_curves()

    class_names = train_dataset.label_encoder.classes_
    utils.plot_confusion_matrix(train_gts, train_preds, classes=class_names, normalize=False,
                                save_to='output/train-conf-mat')
    utils.plot_confusion_matrix(val_gts, val_preds, classes=class_names, normalize=False, save_to='output/val-conf-mat')


def train(frequency, model, train_loader, optimizer, epoch):
    # set model as training mode
    losses = []
    scores = []
    train_labels = []
    train_preds = []

    train_count = 0  # counting total trained sample in one epoch
    for batch_idx, data in enumerate(train_loader):
        X, y, video_ids = data
        # distribute data to device
        X, y = X.cuda(), y.cuda().view(-1, )

        train_count += X.size(0)

        optimizer.zero_grad()
        out = model(X)  # output has dim = (batch, number of classes)

        loss = F.cross_entropy(out, y)


        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(out, 1)[1]  # y_pred != output

        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

        # collect prediction labels
        train_labels.extend(y.cpu().data.squeeze().tolist())
        train_preds.extend(y_pred.cpu().data.squeeze().tolist())

        scores.append(step_score)  # computed on CPU

        loss.backward()


        optimizer.step()

        # show information
        if (batch_idx + 1) % frequency == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.6f}%'.format(
                epoch + 1, train_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * step_score))
            name = datetime.datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
            save_model(model, os.path.join('weights',"model_{}.pt".format(name)))

    return losses, scores, train_labels, train_preds


def validation(model, test_loader, epoch, save_to):

    model.eval()

    val_loss = []
    labels = []
    labels_pred = []
    all_video_ids = []
    all_pool_out = []

    num_copies = 4

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # distribute data to device
            X, y, video_ids = data
            X, y = X.cuda(), y.cuda().view(-1, )

            all_output = []

            stride = X.size()[2] // num_copies

            for i in range(num_copies):
                X_slice = X[:, :, i * stride: (i+1) * stride]
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            output = torch.mean(all_output, dim=1)

           
            loss = F.cross_entropy(output, y)

            val_loss.append(loss.item())  # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            labels.extend(y)
            labels_pred.extend(y_pred)
            all_video_ids.extend(video_ids)
            all_pool_out.extend(output)

    # this computes the average loss on the BATCH
    val_loss = sum(val_loss) / len(val_loss)

    # compute accuracy
    labels = torch.stack(labels, dim=0)
    labels_pred = torch.stack(labels_pred, dim=0).squeeze()
    all_pool_out = torch.stack(all_pool_out, dim=0).cpu().data.numpy()

    # log down incorrectly labelled instances
    incorrect_indices = torch.nonzero(labels - labels_pred).squeeze().data
    incorrect_video_ids = [(vid, int(labels_pred[i].data)) for i, vid in enumerate(all_video_ids) if
                           i in incorrect_indices]

    labels = labels.cpu().data.numpy()
    labels_pred = labels_pred.cpu().data.numpy()

    # top-k accuracy
    top1acc = accuracy_score(labels, labels_pred)
    top3acc = compute_top_n_accuracy(labels, all_pool_out, 3)
    top5acc = compute_top_n_accuracy(labels, all_pool_out, 5)
    top10acc = compute_top_n_accuracy(labels, all_pool_out, 10)
    top30acc = compute_top_n_accuracy(labels, all_pool_out, 30)

    # show information
    print('\nVal. set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(labels), val_loss,
                                                                                        100 * top1acc))

    if save_to:
        # save Pytorch models of best record
        torch.save(model.state_dict(),
                   os.path.join(save_to, 'gcn_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
        print("Epoch {} model saved!".format(epoch + 1))

    return val_loss, [top1acc, top3acc, top5acc, top10acc, top30acc], labels.tolist(), labels_pred.tolist(), incorrect_video_ids



def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    ts = truths
    successes = 0
    for i in range(ts.shape[0]):
        if ts[i] in best_n[i, :]:
            successes += 1
    return float(successes) / ts.shape[0]

def save_model(model, path, test=True):
    '''
    Saves the model at a desired point in time 
    
    Parameters 
    - model should take in a pytorch model 
    - path the path to save the model 
    
    '''
    if test == False:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, path)
    else:
        torch.save(model.state_dict(), path)
        
        
def load_model(path, configs, num_class, test=True):
    '''
    Loads a previously saved model and runs it 
    
    Parameters 
    - path where the saved model is stored
    - test whether the model will be used for testing
    
    '''
    epochs = configs.max_epochs
    log_interval = configs.log_interval
    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages
    model = GCNMultiBlock(input_feature=num_samples*2, hidden_feature=num_samples*2,
                         num_class=num_class, p_dropout=drop_p, num_stage=num_stages).cuda()

    
    if test == False:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()
        
    else:
        model.load_state_dict(torch.load(path))
        model.eval()
    
    return model


if __name__ == "__main__":
    direc = '/home/jovyan/Documents/DL/DL_Project/WLASL'
    subset = 'asl100'
    split_file = os.path.join(direc, 'data/splits/{}.json'.format(subset))
    pose_data = os.path.join(direc, 'data/pose_per_individual_videos')
    config_file = os.path.join(direc, 'sign-language-recognition/{}.ini'.format(subset))
    configs = Config(config_file)
    run(file_split=split_file, configs=configs, pose_directory=pose_data)


