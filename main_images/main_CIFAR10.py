'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import pdb
import os
os.chdir("/scratch/gul15103/main_images")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np

import torchvision
import torchvision.transforms as transforms

#import argparse
from tensorboardX import SummaryWriter

import sys, os
sys.path.append(os.path.join(os.getcwd(), '../'))
sys.path.append(os.path.join(os.getcwd(), './imageNet_models'))
sys.path.append(os.path.join(os.getcwd(), './densenet_pytorch'))
sys.path.append(os.path.join(os.getcwd(), './cifar_models/models'))
from imageNet_models import *
sys.path.append(os.path.join(os.getcwd(), '../optimizer'))
#from utils import progress_bar
import Sadam as Sadam
import SGD_modified as SGD_modified
import adabound as adabound
import densenet_3blocks as dn
import resnet_cifar


import argparse

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--log_interval', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='False', type=bool)
parser.add_argument('--NNtype', '--NeuralNetwork-type', default='DenseNet_BC_100_12', type=str, metavar='DenseNet_BC_100_12', help='Neural Network architecher')
parser.add_argument('--optimizer', default='sgd', type=str, metavar='sgd', help='optimizer')
parser.add_argument('--epsilon', default=1e-8, type=float)
parser.add_argument('--r', default=1, type=int)
parser.add_argument('--reduceLRtype', default='ReduceLROnPlateauMax', type=str)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--partial', default=1/4, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--amsgrad', default=False, type=bool)
parser.add_argument('--transformer', default='log2', type=str)
parser.add_argument('--grad_transf', default='square', type=str)
parser.add_argument('--smooth', default=5, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--hist', default=False, type=bool)
#global args, best_prec
args = parser.parse_args()
best_acc=0

def train_(trainloader, net, device, criterion):
    net.eval()
    train_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
          
    # Save checkpoint.
    return( train_loss/(batch_idx+1), correct/total ) 



def test(testloader, net, device, criterion,epoch,folder_name,optimizer):
    global args 
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
          
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'best_acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        
        #torch.save(state, folder_name+'/ckpt.t7')
        best_acc = acc
    return( test_loss/(batch_idx+1), correct/total ) 


def main(dataset= args.dataset, opt = 'sgd', lr = 0.01,r=1, momentum =0.9, augment=True,test_type = 'Padam', beta2=args.beta2, epsilon=args.epsilon, partial = args.partial, weight_decay= args.weight_decay, amsgrad=args.amsgrad,  transformer=args.transformer, grad_transf='square', smooth=5, hist = False):
    global args
    global best_acc
    # Training settings
    log_interval = args.log_interval
    folder = str( dataset) +'_'+str( args.NNtype)+'_batch_size_'+str(args.b)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data
    print('==> Preparing data..')
    
    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    
    if dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)
    elif dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='../data/CIFAR100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.b, shuffle=True)
        
        testset = torchvision.datasets.CIFAR100(root='../data/CIFAR100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)    
     # ================================================================== #
     #                         Model                                      #
     # ================================================================== #
    print('==> Building model..' + args.NNtype)

    if args.NNtype == 'DenseNet_BC_100_12':
        net=dn.DenseNet3(100, 10, 12, reduction=0.5, bottleneck=True, dropRate=0)
    elif args.NNtype == 'ResNet20': 
        net = resnet_cifar.resnet20_cifar()
    elif args.NNtype == 'ResNet56': 
        net = resnet_cifar.resnet56_cifar()
    elif args.NNtype == 'ResNet18': #for imageNet in original paper
        net = ResNet18()
    elif args.NNtype == 'DenseNet4': #for imageNet in original paper
        net = densenet_cifar()
        
    
    if device == 'cuda':
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    

    
     # ================================================================== #
     #                         Loss and optimizer                         #
     # ================================================================== #    
    
    criterion = nn.CrossEntropyLoss()
    eps=epsilon
    if opt == "Padam":
        if transformer == 'softplus':
            optimizer = Sadam.Sadam(net.parameters(), lr=lr, eps=eps, betas=(0.9, beta2), partial=partial, weight_decay=weight_decay, amsgrad=amsgrad,  transformer= transformer,grad_transf=grad_transf, smooth = smooth, hist= hist)
            folder_name = '../logs_repeat/'+folder+"/"+str(test_type )+"_lr"+str(lr)+'_beta2_'+str(beta2)+'_eps_'+str( eps)+'_'+ str( args.reduceLRtype)+'_wd_'+str( weight_decay)+'_amsgrad_'+str( amsgrad)+str( transformer)+str( grad_transf)+'_smth_'+str( int(smooth))+'_'+str( r )
            file_name = '../logs_repeat/'+folder+"/"+str(test_type )+"_lr"+str(lr)+'_beta2_'+str(beta2)+'_eps_'+str( eps)+'_'+ str( args.reduceLRtype)+'_wd_'+str( weight_decay)+'_amsgrad_'+str( amsgrad)+str( transformer)	+str( grad_transf)+'_smth_'+str( int(smooth))+'_'+str( r )
            
        else:        
            optimizer = Sadam.Sadam(net.parameters(), lr=lr, eps=eps, betas=(0.9, beta2), partial=partial, weight_decay=weight_decay, amsgrad=amsgrad,  transformer= transformer,grad_transf=grad_transf, hist= hist)
            folder_name = '../logs_repeat/'+folder+"/"+str(test_type )+"_lr"+str(lr)+'_beta2_'+str(beta2)+'_eps_'+str( eps)+'_'+ str( args.reduceLRtype)+'_wd_'+str( weight_decay)+'_partial_'+str( partial)+'_amsgrad_'+str( amsgrad)+str( transformer)+str( grad_transf)+'_'+str( r )
            file_name = '../logs_repeat/'+folder+"/"+str(test_type )+"_lr"+str(lr)+'_beta2_'+str(beta2)+'_eps_'+str( eps)+'_'+ str( args.reduceLRtype)+'_wd_'+str( weight_decay)+'_partial_'+str( partial)+'_amsgrad_'+str( amsgrad)+str( transformer)	+str( grad_transf)+'_'+str( r )
    elif opt == "sgd":
        optimizer = SGD_modified.SGD(net.parameters(), lr = lr, momentum=momentum, weight_decay=weight_decay, hist= hist)
        folder_name = '../logs_repeat/'+folder+"/"+str(opt )+"_lr"+str(lr)+"_mom"+str( momentum)+'_'+ str( args.reduceLRtype)+'_wd_'+str( weight_decay)+'_'+str( r )
        file_name = '../logs_repeat/'+folder+"/"+str(opt )+"_lr"+str(lr)+"_mom"+str( momentum)+'_'+ str( args.reduceLRtype)+'_wd_'+str( weight_decay)+'_'+str( r )
    elif opt == "adabound":
        optimizer = adabound.AdaBound(net.parameters(), lr = lr, weight_decay=weight_decay,  amsbound = amsgrad)
        folder_name = '../logs_repeat/'+folder+"/"+str(opt )+"_lr"+str(lr)+'_'+ str( args.reduceLRtype)+'_wd_'+str( weight_decay)+'_amsgrad_'+str( amsgrad)+'_'+str( r )
        file_name = '../logs_repeat/'+folder+"/"+str(opt )+"_lr"+str(lr)+'_'+ str( args.reduceLRtype)+'_wd_'+str( weight_decay)+'_amsgrad_'+str( amsgrad)+'_'+str( r )

    exists = os.path.isfile(file_name+str("_MaxAccuracy"))
    if exists:
        print(file_name+str("_MaxAccuracy has finished"))
        return
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    logger = SummaryWriter(folder_name)
    
    file_1 = open(file_name+str("_percentile.header"),"w")
    file_1.write('epoch,iteration,1%,2.5%,5%,10%,25%,50%,75%,90%,95%,97.5%,99%,min,max,mean,sigma\n') 
    file_1.close()
    
    file_2 = open(file_name+str("_loss.header"),"w")
    file_2.write('epoch,iteration,training_loss,training_accuracy,testing_loss,testing_accuracy\n') 
    file_2.close()
    
    
    maxAccuracy = 0
    
    #if args.resume:
    #    if os.path.isfile(folder_name+'/ckpt.t7'):
     #       print('=> loading checkpoint "{}"'.format(folder_name+'/ckpt.t7'))
     #       checkpoint = torch.load(folder_name+'/ckpt.t7')
     #       args.start_epoch = checkpoint['epoch']
     #       best_acc = checkpoint['best_acc']
     #       net.load_state_dict(checkpoint['net'])
     #       optimizer.load_state_dict(checkpoint['optimizer'])
     #       print("=> loaded checkpoint '{}' (epoch {})".format(folder_name+'/ckpt.t7', checkpoint['epoch']))
     #   else:
     #       print("=> no checkpoint found at '{}'".format(folder_name+'/ckpt.t7'))

     # ================================================================== #
     #                 train model and testing error                      #
     # ================================================================== # 
    
    for epoch in  range(args.start_epoch, args.epochs):
        print('\nEpoch: %d' % epoch)
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            step = epoch*len( trainloader ) + batch_idx 
            if device == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            denom_info= optimizer.step()
    

            if (step+1) % log_interval == 0:
                _, predicted = outputs.max(1)
                accuracy = predicted.eq(targets).sum().item()/targets.size(0)
                
        
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
        
                
                if denom_info and denom_info['m_v_eta']:     
                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in denom_info.items():
                        info_array = torch.cat(value, dim=0).cpu().numpy()
                        logger.add_histogram(tag, info_array, step+1)
                        #pdb.set_trace()
                        temp = np.append( np.array( [epoch+1, step+1 ]), np.percentile( info_array, [1,2.5,5,10,25,50,75,90,95,97.5,99]))
                        temp = np.append( temp, np.array( [np.min( info_array), np.max( info_array)]))
                        
                        mean = np.mean( info_array )
                        sigma = np.std( info_array )
                        
                        temp = np.append( temp, np.array( [mean,  sigma]))
                        #logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step+1)

                        if tag == "denom":
                            if 'denom_array' in locals():
                                denom_array = np.vstack( ( denom_array, temp))
                            else:
                                denom_array = temp
                            
                        elif tag == "denom_inv":
                            if 'denom_inv_array' in locals():
                                denom_inv_array = np.vstack( ( denom_inv_array, temp))
                            else:
                                denom_inv_array = temp
                              
                        elif tag == "m_v_eta":
                            if 'm_v_eta_array' in locals():
                                m_v_eta_array = np.vstack( ( m_v_eta_array, temp))
                            else:
                                m_v_eta_array = temp
                              
                        

        training_loss, training_accuracy=train_( trainloader, net, device, criterion )
        testing_loss, testing_accuracy = test(testloader, net, device, criterion, epoch, folder_name, optimizer)
        
        if testing_accuracy > maxAccuracy:
            maxAccuracy = testing_accuracy
        
        info = {"training_loss": training_loss, 'training_accuracy': training_accuracy,'testing_loss': testing_loss, 'testing_accuracy':testing_accuracy}
        if 'loss_info_array' in locals():
             loss_info_array = np.vstack( ( loss_info_array, np.array( [epoch+1, (epoch+1)*len(trainloader), training_loss, training_accuracy, testing_loss, testing_accuracy])))
        else:
            loss_info_array =  np.array( [epoch+1, (epoch+1)*len(trainloader), training_loss, training_accuracy, testing_loss, testing_accuracy])
        
        for tag, value in info.items():
            logger.add_scalar(tag, value, (epoch+1))

        logger.add_scalar( 'lr', optimizer.param_groups[0]['lr'], epoch+1)

        if args.reduceLRtype == 'ReduceLROnPlateauMax':
          scheduler.step( testing_accuracy)
        elif args.reduceLRtype == 'manual0': 
          if epoch < 150:
             optimizer.param_groups[0]['lr'] = lr
          elif epoch < 225:
             optimizer.param_groups[0]['lr'] = lr * 0.1
          else:
             optimizer.param_groups[0]['lr'] = lr * 0.01


    #pdb.set_trace()                    
    if 'denom_array' in locals():
        np.savetxt(file_name+str("_denom_percentile.info"), denom_array, delimiter=",")
        np.savetxt(file_name+str("_denom_inv_percentile.info"), denom_inv_array, delimiter=",")
    if 'm_v_eta_array' in locals():
        np.savetxt(file_name+str("_m_v_eta_percentile.info"), m_v_eta_array, delimiter=",")

    np.savetxt(file_name+str("_loss.info"), loss_info_array, delimiter=",")
    file = open(file_name+str("_MaxAccuracy"),"w")
    file.write(str( maxAccuracy)+'\n') 
    file.close()
	
if __name__ == '__main__':
    #opts = [  'adam']
    #opts = ['sgd', 'Padam' ]
    opts = [args.optimizer ]
    #lrs = [ 0.001]
    if args.NNtype == 'DenseNet_BC_100_12' or  args.NNtype == 'DenseNet4':
        print( 'here')
        lrs=[ args.lr ]	
        amsgrads =[args.amsgrad]
        
    else:
        lrs=[ args.lr ]
        amsgrads =[True, False]
        
    print( amsgrads )    
    r=1
    #for r in [0]:
    r_s = [6]
    print( lrs)
    for r in r_s:
        for opt in opts:
            for lr in lrs:
                print( "opt : {}, lr : {}".format(opt,  lr))
                if opt == 'sgd':
                    for mom in [0.9]:
                        main(opt =opt, lr = lr,r=r, momentum =mom, augment=True, hist= args.hist)
                elif opt =='adabound':
                     for  amsgrad in amsgrads :
                        main(opt =opt, lr = lr,r=r,  amsgrad=amsgrad, augment=True, hist= args.hist)
                else:
                     for transf in [args.transformer]:
                         for amsgrad in amsgrads :
                             print( 'amsgrad_' + str( amsgrad ))
                             if transf == 'log2' or  transf == 'sigmoid':
                                  main( opt=opt, lr=lr, r=r, amsgrad=amsgrad, transformer=transf, augment=True,   grad_transf= args.grad_transf, hist=args.hist)
                             elif transf == 'softplus':
                                 for smooth in [  args.smooth]:
                                     print( smooth)
                                     for grad_transf in [args.grad_transf]:
                                         main( opt=opt, lr=lr, r=r, amsgrad=amsgrad, transformer=transf, augment=True,   grad_transf= grad_transf, smooth = smooth, hist=args.hist)
                             elif transf == 'Padam':
                                 print('start')
                                 for par in [args.partial]:
                                     main( opt=opt, lr=lr, r=r, partial=par, amsgrad=amsgrad, transformer=transf, augment=True,    grad_transf=args.grad_transf, hist=args.hist)
     
 