import torch
from tqdm import tqdm

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, y_device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for (x1,x2,x3,x4,label) in tqdm(iterator):
        
        label = label.to(y_device)
        optimizer.zero_grad()
        fx = model(x1,x2,x3,x4)
        loss = criterion(fx, label)
        acc = calculate_accuracy(fx, label)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, y_device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for (x1,x2,x3,x4,label) in tqdm(iterator):
            label = label.to(y_device)
            fx = model(x1,x2,x3,x4)

            loss = criterion(fx, label)
            acc = calculate_accuracy(fx, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)