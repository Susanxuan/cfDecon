import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import simdatset, AutoEncoder, device
from utils import showloss, score

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train(model, train_loader, optimizer, epochs=1):
    model.train()
    loss = []
    recon_loss = []
    for i in tqdm(range(epochs)):
        for k, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)
            # print('x_recon',x_recon)
            # print('data',data)
            # print('x_recon_loss', F.l1_loss(x_recon,data))
            batch_loss = 100*F.l1_loss(cell_prop, label)+F.l1_loss(x_recon,data)
            batch_loss.backward()
            optimizer.step()
            loss.append(F.l1_loss(cell_prop, label).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

    return model, loss, recon_loss

def train2(model, train_loader, optimizer, epochs=1):
    # two stage training
    model.train()
    loss = []
    recon_loss = []
    
    # Stage 1: Train the Encoder
    for i in tqdm(range(epochs)):
        for k, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            _, cell_prop, _ = model(data)
            batch_loss = F.l1_loss(cell_prop, label)
            batch_loss.backward()
            optimizer.step()
            loss.append(batch_loss.cpu().detach().numpy())
    
    # Stage 2: Fix the Encoder and Train the Decoder
    for i in tqdm(range(epochs)):
        for k, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            x_recon, _, _ = model(data)
            batch_loss = F.l1_loss(x_recon, data)
            batch_loss.backward()
            optimizer.step()
            recon_loss.append(batch_loss.cpu().detach().numpy())

    return model, loss, recon_loss

def adaptive_stage(model, data, optimizerD, optimizerE, step=10, max_iter=5):
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    loss = []
    model.eval()
    model.state = 'test'
    _, ori_pred, ori_sigm = model(data)
    ori_sigm = ori_sigm.detach()
    ori_pred = ori_pred.detach()
    model.state = 'train'

    for k in range(max_iter):
        model.train()
        for i in range(step):
            reproducibility(seed=0)
            optimizerD.zero_grad()
            x_recon, _, sigm = model(data)
            batch_loss = F.l1_loss(x_recon, data)+F.l1_loss(sigm,ori_sigm)
            batch_loss.backward()
            optimizerD.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

        for i in range(step):
            reproducibility(seed=0)
            optimizerE.zero_grad()
            x_recon, pred, _ = model(data)
            batch_loss = F.l1_loss(ori_pred, pred)+F.l1_loss(x_recon, data)
            batch_loss.backward()
            optimizerE.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

    model.eval()
    model.state = 'test'
    _, pred, sigm = model(data)
    return sigm.cpu().detach().numpy(), loss, pred.detach().cpu().numpy()

def predict(test_x, genename, celltypes, samplename,
            model_name=None, model=None,
            adaptive=True, mode='high-resolution'):
    
    if model is not None and model_name is None:
        print('Model is saved without defined name')
        torch.save(model, 'model.pth')
    if adaptive is True:
        if mode == 'high-resolution':
            TestSigmList = np.zeros((test_x.shape[0], len(celltypes), len(genename), 5))
            TestPred = np.zeros((test_x.shape[0], len(celltypes)))
            print('Start adaptive training at high-resolution')
            for i in tqdm(range(len(test_x))):
                x = test_x[i,:,:].reshape(1,test_x.shape[1],test_x.shape[2])
                # print(x.shape)
                if model_name is not None and model is None:
                    model = torch.load(model_name + ".pth")
                elif model is not None and model_name is None:
                    model = torch.load("model.pth")
                decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
                encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
                optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
                optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
                test_sigm, loss, test_pred = adaptive_stage(model, x, optimizerD, optimizerE, step=300, max_iter=3)
                # print('TestPred',TestPred.shape,'test_pred',test_pred.shape,'TestPred[i,:]',TestPred[i,:].shape)
                TestPred[i,:] = test_pred
                # print('test_sigm',test_sigm.shape,'TestSigmList',TestSigmList.shape,'TestSigmList[i, :, :,:]',TestSigmList[i, :, :,:].shape)
                TestSigmList[i, :, :,:] = test_sigm
                
            # TestPred = pd.DataFrame(TestPred,columns=celltypes,index=samplename)
            CellTypeSigm = {}
            for i in range(len(celltypes)):
                cellname = celltypes[i]
                sigm = TestSigmList[:,i,:,:]
                # sigm = pd.DataFrame(sigm,columns=genename,index=samplename)
                CellTypeSigm[cellname] = sigm
            print('Adaptive stage is done')

            return TestPred,CellTypeSigm

        elif mode == 'overall':
            if model_name is not None and model is None:
                model = torch.load(model_name + ".pth")
            elif model is not None and model_name is None:
                model = torch.load("model.pth")
            decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
            encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
            optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
            optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
            print('Start adaptive training for all the samples')
            test_sigm, loss, test_pred = adaptive_stage(model, test_x, optimizerD, optimizerE, step=300, max_iter=3)
            showloss(loss,'/mnt/nas/user/yixuan/cfDNA/CelFEER/output/Multi_channel_WGBS/two_sim/adap_loss.png')
            print('Adaptive stage is done')
            # test_pred = pd.DataFrame(test_pred,columns=celltypes,index=samplename)
            # test_sigm = pd.DataFrame(test_sigm,columns=genename,index=celltypes)
            return test_pred, test_sigm 

    else:
        if model_name is not None and model is None:
            model = torch.load(model_name+".pth")
        elif model is not None and model_name is None:
            model = model
        print('Predict cell fractions without adaptive training')
        model.eval()
        model.state = 'test'
        data = torch.as_tensor(test_x, dtype=torch.float32, device=device)
        _, pred, _ = model(data)
        pred = pred.cpu().detach().numpy()
        # pred = pd.DataFrame(pred, columns=celltypes, index=samplename)
        print('Prediction is done')
        return pred
    

def test_AE_function(train_x,train_y,test_x,test_y,batch_size=128,outdir=None,adaptive=True, mode='overall'):
    reproducibility(seed=0)
    train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)
    model = AutoEncoder(train_x.shape[1], train_x.shape[2], train_y.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3) #1e-4
    model, loss, reconloss = train(model, train_loader, optimizer, epochs=int(5000/(len(train_x)/batch_size)))
    showloss(loss,outdir+'/loss.png')
    showloss(reconloss,outdir+'/reconloss.png')
    genename=range(test_x.shape[1])
    celltypes=range(test_y.shape[1])
    samplename=range(test_y.shape[0])
    if adaptive:
        pred, train_sigm = predict(test_x=test_x, genename=genename, celltypes=celltypes, samplename=samplename,
                        model=model,adaptive=adaptive, mode=mode)
        l1, ccc = score(pred,test_y.cpu().detach().numpy())
        return l1, ccc, pred, train_sigm
    else:
        pred = predict(test_x=test_x, genename=genename, celltypes=celltypes, samplename=samplename,
                        model=model,adaptive=adaptive, mode=mode)
        l1, ccc = score(pred,test_y.cpu().detach().numpy())
        return l1, ccc, pred