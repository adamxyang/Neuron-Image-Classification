import torch
from tqdm import tqdm

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc


def train(model, device, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for (x, label) in tqdm(iterator):
        x = x.to(device, dtype=torch.float)
        label = label.to(device)
        
        optimizer.zero_grad()
        fx = model(x)
        loss = criterion(fx, label)
        acc = calculate_accuracy(fx, label)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, device, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for (x, label) in tqdm(iterator):
            x = x.to(device, dtype=torch.float)
            label = label.to(device)
            fx = model(x)

            loss = criterion(fx, label)
            acc = calculate_accuracy(fx, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
