import torch
from torch import optim
import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from models.train_eval import train, evaluate
import torch.nn as nn
from sklearn.metrics import classification_report
device = torch.device('cpu')
def trainer(model, train_dl, test_dl, data_id, config, params):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    target_names = ['Healthy','D1','D2','D3','D4','D5','D6','D7','D8','D9','D10']
    for epoch in range(params['pretrain_epoch']):
        start_time = time.time()
        train_loss, train_pred, train_labels = train(model, train_dl, optimizer, criterion, config)
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        # Evaluate on the test set
        test_loss,_, _= evaluate(model, test_dl, criterion, config)
        print('=' * 50)
        print(f'\tTest Loss: {test_loss:.3f}')
        train_labels = torch.stack(train_labels).view(-1)
        train_pred = torch.stack(train_pred).view(-1)
        print(classification_report(train_labels, train_pred, target_names=target_names))
    # Evaluate on the test set
    test_loss, y_pred, y_true = evaluate(model, test_dl, criterion, config)
    y_true = torch.stack(y_true).view(-1)
    y_pred = torch.stack(y_pred).view(-1)
    print('=' * 50)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} ')#| Score: {test_score:7.3f}')
    print('=' * 50)
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(confusion_matrix(y_true,y_pred))
    ##### Plotting the confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    ##### Compute confusion matrix
    import itertools 
    cnf_matrix = (confusion_matrix(y_true, y_pred))
    np.set_printoptions(precision=2)
    # Plot normalized confusion matrix
    plt.figure()
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix')
    plt.show()    
    print('The classification accuracy is =', accuracy_score(y_true, y_pred , normalize=True))
    print('The f1 score is = ', f1_score(y_true, y_pred, average='weighted'))
    print('The MCC Coefficient is =', matthews_corrcoef(y_true, y_pred))
    # print('The ROC_AUC scrore is', roc_auc_score(y_true, y_scores))
    print('| End of Pre-training  |')
    print('=' * 50)
    return model    
