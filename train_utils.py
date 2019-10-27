import torch
from tqdm import tqdm
import CLR as CLR
import OneCycle as OneCycle

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


def update_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def update_mom(optimizer, mom):
    for g in optimizer.param_groups:
        g['momentum'] = mom


        

def train(model, device, iterator, criterion, onecycle, optimizer):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for (x, label) in tqdm(iterator):
        x = x.to(device)
        label = label.to(device)
        
        lr, mom = onecycle.calc()
        update_lr(optimizer, lr)
        update_mom(optimizer, mom)
        
        optimizer.zero_grad()
        fx = model(x)
        loss = criterion(fx, label)
        acc = calculate_accuracy(fx, label)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), onecycle

def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for (x, label) in tqdm(iterator):
            x = x.to(device)
            label = label.to(device)
            fx = model(x)

            loss = criterion(fx, label)
            acc = calculate_accuracy(fx, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
