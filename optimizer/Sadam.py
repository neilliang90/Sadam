from torch.optim import *
import torch
import math
import numpy as np
import pdb 

class Sadam(Optimizer):
    """Implements Sadam/Samsgrad algorithm.
		It has been proposed in `Calibrating the Learning Rate for Adaptive Gradient Methods to Improve Generalization Performance`_.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-1)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        partial (float, optional): partially adaptive parameter
    """

    def __init__(self, params, lr=1e-1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=True, partial = 1/4, transformer='Padam', test_type='Padam', hist=False, grad_transf = 'square', smooth = 50):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, partial = partial, transformer=transformer, test_type=test_type, hist=hist, grad_transf = grad_transf , smooth = smooth)
        super(Padam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        denom_list = []
        denom_inv_list = []
        m_v_eta = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                partial = group['partial']
                grad_transf = group['grad_transf']
                smooth = group['smooth']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                #pdb.set_trace()
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                
                ################ tranformer for grad ###########################################################    
                if grad_transf == 'square':
                    grad_tmp = grad**2
                elif grad_transf == 'abs':
                    grad_tmp = grad.abs()
                    
                    
                if group['test_type'] ==  "adam_test1":
                    temp_1 = torch.exp(torch.sign( grad_tmp-exp_avg_sq))
                    exp_avg_sq.mul_(1).addcmul_(beta2**state['step'], temp_1, grad_tmp)
                elif group['test_type'] ==  "adam_test2":
                    #exp_avg_sq.mul_(1).addcmul_(beta2**state['step'], grad, grad)
                    exp_avg_sq.mul_(1).add_(beta2**state['step']* grad_tmp)
                else:
                    exp_avg_sq.mul_(beta2).add_((1 - beta2)*grad_tmp)
                    
                    
                    
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.clone()
                else:
                    denom = exp_avg_sq.clone()

                if grad_transf == 'square':
                    #pdb.set_trace()
                    denom.sqrt_() 
 
                
            
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                ################ tranformer for denom ###########################################################        
                if  group['transformer'] =='softplus':
                    sp = torch.nn.Softplus( smooth)
                    denom = sp( denom)
                    p.data.addcdiv_(-step_size, exp_avg, denom )                    
                else:
                    p.data.addcdiv_(-step_size, exp_avg, (denom.add_(group['eps']))**(partial*2))
 
    
                ################ logger ###########################################################       
                if group['hist']:    
                    if group['transformer'] =='log2':
                        denom_list.append((torch.log2( denom+1) + group['eps']).reshape( 1, -1).squeeze())
                        denom_inv_list.append( (1/(torch.log2( denom+1) + group['eps'])).reshape( 1, -1).squeeze() )
                        m_v_eta.append( (-step_size*torch.mul(exp_avg, (1/(torch.log2( denom+1) + group['eps']) ))).reshape( 1, -1).squeeze()  )
                    elif group['transformer'] =='sigmoid'or  group['transformer'] =='softplus':
                        denom_list.append(denom.reshape( 1, -1).squeeze())
                        denom_inv_list.append( (1/denom).reshape( 1, -1).squeeze() )
                        m_v_eta.append( (-step_size*torch.mul(exp_avg, (1/denom ))).reshape( 1, -1).squeeze()  )
                    else:
                        denom_list.append(denom.reshape( 1, -1).squeeze())
                        denom_inv_list.append( (1/(denom)**(partial*2)).reshape( 1, -1).squeeze() ) 
                        m_v_eta.append( (-step_size*torch.mul(exp_avg, (1/(denom)**(partial*2) ))).reshape( 1, -1).squeeze()  )
                            
                        
        return {"denom": denom_list,  "denom_inv": denom_inv_list, "m_v_eta":m_v_eta }

    
