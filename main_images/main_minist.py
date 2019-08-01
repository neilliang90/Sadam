import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import sys, os
sys.path.append(os.path.join(os.getcwd(), '../optimizer'))
import adampp as adampp
import adam_test1 as adam_test1
from torchvision import datasets, transforms
import torch.nn.functional as F
import argparse
import Sadam as Sadam

#Architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        #self.conv1 = nn.Conv2d(1, 20, 5)
        #self.conv2 = nn.Conv2d(20, 50, 5)
        #self.fc1 = nn.Linear(4*4*50, 500)
        #self.fc2 = nn.Linear(500, 10)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16*5*5, 120)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)     
        #x = x.view(-1, 4*4*50)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        #x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

net = NeuralNet()
save_net = net

# Test 
def test(model, device, test_loader,loss_function):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss  += loss_function(output, target)*len( data) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)


    return test_loss, test_accuracy


def main(opt = 'sgd', lr = 0.05,eps=1e-8, momentum =0, r= 0, partial=1/4, weight_decay=0.0001, amsgrad=True, transformer='linear',  test_type='Padam', hist=True, grad_transf = 'square'):
    # Training settings
    epochs =40
    log_interval=100


    folder = 'mnist'
    #torch.manual_seed(args.seed)


    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MNIST dataset    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True)
    
    
    net = NeuralNet().to(device)
    
    
    
    # Loss and optimizer
    loss_function = nn.CrossEntropyLoss()  
    if opt == "Padam":
        optimizer = Sadam.Sadam(net.parameters(), lr=lr, eps=eps, partial=partial, weight_decay=weight_decay, amsgrad=amsgrad,  transformer= transformer, test_type=test_type, hist=hist, grad_transf = grad_transf)
        if transformer == 'Padam':
            folder_name = '../logs/'+folder+"/"+str(test_type )+"_lr"+str(lr)+'_eps_'+str( eps)+'_weight_decay_'+str( weight_decay)+'_partial_'+str( partial)+'_amsgrad_'+str( amsgrad)+'_denom_transformer_'+str( transformer)+'_grad_transf_'+str( grad_transf)+'_'+str( r )
            file_name = "../logs/"+folder+"/"+str(test_type )+"_lr"+str(lr)+'_eps_'+str( eps)+'_weight_decay_'+str( weight_decay)+'_partial_'+str( partial)+'_amsgrad_'+str( amsgrad)	+'_denom_transformer_'+str( transformer)+'_grad_transf_'+str( grad_transf)+'_'+str( r )
        else:
            folder_name = '../logs/'+folder+"/"+str(test_type )+"_lr"+str(lr)+'_eps_'+str( eps)+'_weight_decay_'+str( weight_decay)+'_amsgrad_'+str( amsgrad)+'_denom_transformer_'+str( transformer)+'_grad_transf_'+str( grad_transf)+'_'+str( r )
            file_name = "../logs/"+folder+"/"+str(test_type )+"_lr"+str(lr)+'_eps_'+str( eps)+'_weight_decay_'+str( weight_decay)+'_amsgrad_'+str( amsgrad)	+'_denom_transformer_'+str( transformer)+'_grad_transf_'+str( grad_transf)+'_'+str( r )
                
    elif opt == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=momentum, weight_decay=weight_decay)
        folder_name = '../logs/'+folder+"/sgd_lr"+str(lr)+"_mom"+str( momentum)+'_'+str( r )+'_weight_decay_'+str( weight_decay)
        file_name = "../logs/"+folder+"sgd_lr"+str(lr)+"_mom"+str( momentum)+'_'+str( r )+'_weight_decay_'+str( weight_decay)
	
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.1, 10*len(train_loader)/log_interval)
    logger = SummaryWriter(folder_name)
    file = open(file_name+str(".txt"),"w")
    file.write('epoch,iteration,testing_loss,testing_accuracy\n') 
    
    
    
    #train
    data_iter = iter(train_loader)
    iter_per_epoch = len(train_loader)
    total_step =epochs*iter_per_epoch
    
    # Start training
    for step in range(total_step):
        
        # Reset the data_iter
        if (step+1) % iter_per_epoch == 0:
            data_iter = iter(train_loader)
    
        # Fetch images and labels
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = net(images)
        loss = loss_function(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        denom_info= optimizer.step()
    
        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()
    
        if (step+1) % log_interval == 0:
            print ('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}' 
                   .format(step+1, total_step, loss.item(), accuracy.item()))
    
            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
    
            # 1. Log scalar values (scalar summary)
            info = { 'training_loss': loss.item(), 'training_accuracy': accuracy.item() }
    
            for tag, value in info.items():
                logger.add_scalar(tag, value, step+1)
            
            testing_loss, testing_accuracy = test( net, device, test_loader,loss_function)
            logger.add_scalar( 'lr', optimizer.param_groups[0]['lr'], step+1)
            scheduler.step( accuracy)
            
            info = { 'testing_loss': testing_loss, 'testing_accuracy': testing_accuracy }
    
            for tag, value in info.items():
                logger.add_scalar(tag, value, step+1)
            
            
            file.write(str((step+1)/log_interval) +','+str(step+1)+','+ str(testing_loss)+','+str(testing_accuracy)+'\n') 
                
 
            if denom_info and denom_info['denom']: 
                for tag, value in denom_info.items():
                    logger.add_histogram(tag, torch.cat(value, dim=0).cpu().numpy(), step+1)
                    #logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step+1)
                   
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.add_histogram(tag, value.data.cpu().numpy(), step+1)
                    logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), step+1)
    file.close()
        
if __name__ == '__main__':
    opts = [ 'Padam']#,'sgd']
    lrs = [   0.1, 0.01, 0.001]
    for r in range( 1 ):
        for opt in opts:
            for lr in lrs:
                print( "opt : {}, lr : {}".format(opt,  lr))
                if opt == 'sgd':
                    for mom in [0.9]:
                        main( opt , lr, mom, r)
                else:
                     for transf in ['sigmoid']
                         for amsgrad in [False, True]:
                             if  transf == 'softplus':
                                  main( opt=opt, lr=lr, r=r,  amsgrad=amsgrad, transformer=transf,  test_type='Padam', hist=True, grad_transf='square')
                             elif transf == 'Padam':
                                 for par in [1/4, 1/8, 1/16]:
                                     for eps in [1e-3, 1e-8]:
                                         main( opt=opt, lr=lr, eps=eps, r=r, partial=par, amsgrad=amsgrad, transformer=transf,  test_type='Padam', hist=True, grad_transf='abs')
      