import argparse
import os
from threading import Lock
import time
import datetime
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import random
from random import Random
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import *
import copy
import math
lock = Lock()
import argparse
import os
from threading import Lock
import time
import datetime
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import random
from random import Random
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import *
import copy
import math
from threading import Timer
lock = Lock()
parser = argparse.ArgumentParser(
    description="Parameter-Server RPC based training")
parser.add_argument(
    "--world_size",
    type=int,
    default=3,
    help="""Total number of participating processes. Should be the sum of
       master node and all training nodes.""")
parser.add_argument("--train_type", type=str, default='H_local')
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--grad_value", type=float, default=0.5)
parser.add_argument("--H", type=int, default=2)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--learning_rate2", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--warmup_epoch", type=int, default=5)
parser.add_argument("--compress_ratio", type=float, default=1.0)
parser.add_argument("--sample_ratio", type=float, default=1.0)
parser.add_argument("--data_worker", type=int, default=1)
parser.add_argument("--epochs",type=int,default=800)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=200,
    help="""Total number of training time.""")
parser.add_argument(
    "--rank",
    type=int,
    default=0,
    help="Global rank of this process. Pass in 0 for master.")
parser.add_argument(
    "--num_gpus",
    type=int,
    default=0,
    help="""Number of GPUs to use for training, currently supports between 0
        and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
parser.add_argument(
    "--master_addr",
    type=str,
    default="172.18.0.2",
    help="""Address of master, will default to localhost if not provided.
       Master must be able to accept network traffic on the address + port.""")
parser.add_argument(
    "--master_port",
    type=str,
    default="29500",
    help="""Port that master is listening on, will default to 29500 if not
       provided. Master must be able to accept network traffic on the host and port.""")

args = parser.parse_args()
grad_value = args.grad_value
H = args.H
compress_ratio = args.compress_ratio
sample_ratio = args.sample_ratio
world_size = args.world_size
train_type = args.train_type
learning_rate = args.learning_rate
learning_rate2 = args.learning_rate2
batch_size = args.batch_size
warmup_epoch = args.warmup_epoch
max_epoch = args.n_epochs
epoch2=args.epochs
device = args.device
data_worker = args.data_worker
type_list = ['BSP', 'SSP', 'ASP']

# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.
def call_method(method, rref, *args, **kwargs):
    # print('call_method')
    return method(rref.local_value(), *args, **kwargs)


# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.


def remote_method(method, rref, *args, **kwargs):
    # print('remote_method')
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


def write_log(str_log, client_id):  # 写日志函数
    f = './log/' + str(client_id) + ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(str(str_log) + "\n")
        file.close()


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------- Parameter Server --------------------
class ParameterServer(nn.Module):

    def __init__(self):
        super().__init__()
        self.data_dict = {}
        self.client_num = 0
        self.fast = 0
        self.lowest = 0
        self.model = VGG('VGG16')
        self.test_model = VGG('VGG16').cuda()
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)
        warm_up_with_cosine_lr = lambda epoch: epoch / warmup_epoch if epoch <= warmup_epoch else 0.5 * (
                math.cos((epoch - warmup_epoch) / (max_epoch - warmup_epoch) * math.pi) + 1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_cosine_lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size =50, gamma = 0.1, last_epoch=-1)
        self.waiting_time = {}
        self.desparsify_time = {}
        self.ps_time = {}
        self.all_time = {}
        self.wait_epoch = []
        self.monitor = []
        self.monitor1 = []
        self.monitor2 = []
        self.new_epoch = 0
        self.mean_tensor_list = {}
        self.ModelProcessing = None
        self.wait_client_sc = world_size - 1
        self.client_dict = {}
        for i in range(self.wait_client_sc):
            self.client_dict[i]=0
        self.finish_node = 0
        self.curr_update_size = 0
        self.sync_layer_list = []
        self.flag = True
        self.count = 0
        self.c_dict = {}
        self.m_dict = {}
        self.part = []
        self.H = []
        self.mean_dic = {}
        self.global_start_time=0
        self.test_num = 0
        self.continue_flag = True
        self.change_flag =False
        self.continue_num = 0
        self.global_para= {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def desparsify(self, values, indices, ctx):
        #values, indices = tensors
        numel, shape, name = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices.long(), values)
        #print("after shape,",indices.long().dtype())
        return tensor_decompressed.view(shape)
        
    def pre_train(self, args, dic):
        time2 = time.time()
        ps_time1 = time2 - args[2]
        self.flag = True
        client_id = args[0]-1
        step = args[1]
        self.client_dict[client_id] = step
        # findmin = [self.client_dict[u] for u in range(self.wait_client_sc)]
        # while (step - 3) > (min(findmin)):
        #     time.sleep(0.001) 
        #     findmin = [self.client_dict[u] for u in range(self.wait_client_sc)] 
        ps_time = [ps_time1, time.time()]
        return ps_time,dic
    
    def pre_train1(self, args, dic):
        time2 = time.time()
        ps_time1 = time2 - dic['time']
        client_id = dic['client'] - 1
        step = dic['step']      
        self.client_dict[client_id] = step
        lock.acquire()
        time1=time.time()
        degradients = {}
        for tensors in args:
            values, indices, ctx  = tensors
            numel, shape, name = ctx
            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
            tensor_decompressed.scatter_(0, indices.long(), values)
            afterdesparsify=tensor_decompressed.view(shape)
            degradients[name] = afterdesparsify
        time1=time.time()-time1
        str_log = str(dic['client']) + ',' + str(step)+ ',' + str(time1)
        write_log(str_log,"server1")
        for k in self.global_para.keys():
            self.global_para[k].add_(degradients[k].mul_(0.1))
        self.model.load_state_dict(self.global_para, strict=False)    
        self.global_para = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        global_para ={k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        lock.release()
        ps_time = [ps_time1, time.time()]
        return ps_time,global_para   

    def get_h(self):
        n = self.wait_client_sc
        list1 = [self.c_dict[i]  for i in range(self.wait_client_sc)]
        list2 = [self.m_dict[i]  for i in range(self.wait_client_sc)]
        list3 = [list1[i]+list2[i]  for i in range(self.wait_client_sc)]
        x_max = max(list3)
        h_list = []
        h_list1 = []
        p_list = []
        sum_list = [0 for i in range(n)]
        sum1 = x_max
        sum_waiting_time = 0
        sum_waiting_time1 = 0
        sum_waiting_time2 = 0
        k=2
        list3 = [list1[i]*k +list2[i] for i in range(self.wait_client_sc)]
        x_max = max(list3)
        h_list2 = [0 for i in range(n)]
        for j in range(0, n):
            h_list2[j] = round((x_max-list2[j])/list1[j])
            sum_list[j] = h_list2[j]*list1[j]+list2[j]
        if max(sum_list)-min(sum_list)<sum1:
            sum1 = max(sum_list)-min(sum_list)
            sum_waiting_time1 = sum1*600
            h_list1 =[k for k in h_list2]
        h_list.append(h_list1)
        sum_h=sum(h_list1)
        part=[float(h_list1[i])/sum_h for i in range(self.wait_client_sc)]
        part=[round(part[i],3)for i in range(self.wait_client_sc)]
        sum1 = max(list3)
        k=4
        list3 = [list1[i]*k +list2[i] for i in range(self.wait_client_sc)]
        x_max = max(list3)
        h_list2 = [0 for i in range(n)]
        for j in range(0, n):
            h_list2[j] = round((x_max-list2[j])/list1[j])
            sum_list[j] = h_list2[j]*list1[j]+list2[j]
            sum1 = max(sum_list)-min(sum_list)
            sum_waiting_time2 = sum1*600
            h_list1 =[k for k in h_list2]
        sum_waiting_time =sum_waiting_time1+sum_waiting_time2
        h_list.append(h_list1)
        k=8
        list3 = [list1[i]*k +list2[i] for i in range(self.wait_client_sc)]
        x_max = max(list3)
        h_list2 = [0 for i in range(n)]
        for j in range(0, n):
            h_list2[j] = round((x_max-list2[j])/list1[j])
            sum_list[j] = h_list2[j]*list1[j]+list2[j]
            sum1 = max(sum_list)-min(sum_list)
            sum_waiting_time2 = sum1*600
            h_list1 =[k for k in h_list2]
        sum_waiting_time =sum_waiting_time1+sum_waiting_time2
        h_list.append(h_list1)
        k=16
        list3 = [list1[i]*k +list2[i] for i in range(self.wait_client_sc)]
        x_max = max(list3)
        h_list2 = [0 for i in range(n)]
        for j in range(0, n):
            h_list2[j] = int((x_max-list2[j])/list1[j])
            sum_list[j] = h_list2[j]*list1[j]+list2[j]
            sum1 = max(sum_list)-min(sum_list)
            sum_waiting_time2 = sum1*600
            h_list1 =[k for k in h_list2]
        sum_waiting_time =sum_waiting_time1+sum_waiting_time2
        h_list.append(h_list1)
        h_list2 = [k*3 for k in h_list1]
        h_list.append(h_list2)
        h_list2 = [k*4 for k in h_list1]
        h_list.append(h_list2)
        h_list2 = [k*5 for k in h_list1]
        h_list.append(h_list2)
        print("waiting time: ",sum_waiting_time)
        print(h_list)
        print(len(h_list))
        p_list=part
        return h_list, p_list

    # def get_h(self):
    #     n = self.wait_client_sc
    #     list1 = [self.c_dict[i]  for i in range(self.wait_client_sc)]
    #     list2 = [self.m_dict[i]  for i in range(self.wait_client_sc)]
    #     list3 = [list1[i]+list2[i]  for i in range(self.wait_client_sc)]
    #     x_max = max(list3)
    #     h_list = []
    #     h_list1 = []
    #     p_list = []
    #     sum_list = [0 for i in range(n)]
    #     sum1 = x_max
    #     sum_waiting_time = 0
    #     sum_waiting_time1 = 0
    #     sum_waiting_time2 = 0
    #     for k in range(2, 5):
    #         list3 = [list1[i]*k +list2[i] for i in range(self.wait_client_sc)]
    #         x_max = max(list3)
    #         h_list2 = [0 for i in range(n)]
    #         for j in range(0, n):
    #             h_list2[j] = int((x_max-list2[j])/list1[j])
    #             sum_list[j] = h_list2[j]*list1[j]+list2[j]
    #         if max(sum_list)-min(sum_list)<sum1:
    #             sum1 = max(sum_list)-min(sum_list)
    #             sum_waiting_time1 = sum1*600
    #             h_list1 =[k for k in h_list2]
    #     h_list.append(h_list1)
    #     sum_h=sum(h_list1)
    #     part=[float(h_list1[i])/sum_h for i in range(self.wait_client_sc)]
    #     part=[round(part[i],3)for i in range(self.wait_client_sc)]
    #     sum1 = max(list3)
    #     if min(h_list1)==2:
    #         k= 3*min(h_list1)
    #     else:
    #         k= 2*min(h_list1)
    #     list3 = [list1[i]*k +list2[i] for i in range(self.wait_client_sc)]
    #     x_max = max(list3)
    #     h_list2 = [0 for i in range(n)]
    #     for j in range(0, n):
    #         h_list2[j] = int((x_max-list2[j])/list1[j])
    #         sum_list[j] = h_list2[j]*list1[j]+list2[j]
    #         sum1 = max(sum_list)-min(sum_list)
    #         sum_waiting_time2 = sum1*600
    #         h_list1 =[k for k in h_list2]
    #     sum_waiting_time =sum_waiting_time1+sum_waiting_time2
    #     h_list.append(h_list1)
    #     h_list2 = [k*3 for k in h_list1]
    #     h_list.append(h_list2)
    #     h_list2 = [k*4 for k in h_list1]
    #     h_list.append(h_list2)
    #     h_list2 = [k*5 for k in h_list1]
    #     h_list.append(h_list2)
    #     print("waiting time: ",sum_waiting_time)
    #     print(h_list)
    #     print(len(h_list))
    #     p_list=part
    #     return h_list, p_list

    def get_part(self, args):
        time1 = time.time()
        self.flag = True
        self.count += 1
        client_id = args[0]-1
        self.c_dict[client_id] = args[1]
        self.m_dict[client_id] = args[2]+args[3]
        while self.count < self.wait_client_sc:
            time.sleep(0.01)
        lock.acquire()
        if self.flag:
            self.flag = False
            self.H ,self.part = self.get_h()
        lock.release()
        return self.H ,self.part 
    
    def SLocal(self, args, dic):
        # print('BSP')
        flag1 = True
        time0 = time.time()
        self.flag = True
        # print('dic',dic)
        client_id = dic['client'] - 1
        step = dic['step']
        endflag = dic['end_flag']
        degradients = {}
        for tensors in args:
            values, indices, ctx  = tensors
            numel, shape, name = ctx
            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
            tensor_decompressed.scatter_(0, indices.long(), values)
            afterdesparsify=tensor_decompressed.view(shape)
            degradients[name] = afterdesparsify
        self.data_dict[client_id] = degradients
        self.desparsify_time[client_id] = time.time()-time0
        time1 = time.time()
        self.client_dict[client_id] = step
        findmin = [self.client_dict[u] for u in range(self.wait_client_sc)]
        while (step - 3) > (min(findmin)):
            time.sleep(0.001) 
            findmin = [self.client_dict[u] for u in range(self.wait_client_sc)] 
        if self.finish_node!=0 or not self.continue_flag:
            w_avg = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
            flag1 = False
            time1 = time.time()-time1
            return time1,w_avg,flag1       
        time2 = time.time() - time1
        self.waiting_time[client_id] += time2
        time3 = time.time()
        lock.acquire()
        for k in self.global_para.keys():
            self.global_para[k].add_(degradients[k].mul_(self.part[client_id]))
        self.model.load_state_dict(self.global_para, strict=False)    
        self.global_para = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        global_para ={k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        lock.release()
        time4 = time.time() - time3
        self.ps_time[client_id] += time4
        self.all_time[client_id] += (time.time() - time0)
        a = client_id + 1
        str_log = str(a) + ',' + str(step) + ',' + str(self.all_time[client_id])  + ',' +  str(self.desparsify_time[client_id])+ ',' + str(
                self.waiting_time[client_id]) + ',' + str(self.ps_time[client_id])
        self.wait_epoch.append(str_log)
        self.all_time[client_id] = 0
        self.waiting_time[client_id] = 0
        self.ps_time[client_id] = 0
        time5 = time.time() - time0
        return time5, global_para,flag1

    def ALocal(self, args, dic):
        # print('BSP')
        flag1 = True
        time1 = time.time()
        self.flag = True
        # print('dic',dic)
        client_id = args[0] - 1
        step = args[1]
        endflag = args[2]
        self.client_dict[client_id] = step
        if self.finish_node!=0 or not self.continue_flag:
            w_avg = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
            flag1 = False
            time1 = time.time()-time1
            return time1,w_avg,flag1       
        time2 = time.time() - time1
        self.waiting_time[client_id] += time2
        time3 = time.time()
        lock.acquire()
        for k in self.global_para.keys():
            self.global_para[k].add_(dic[k].mul_(self.part[client_id]))
        self.model.load_state_dict(self.global_para, strict=False)    
        self.global_para = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        global_para ={k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        lock.release()
        time4 = time.time() - time3
        self.ps_time[client_id] += time4
        self.all_time[client_id] += (time.time() - time1)
        a = client_id + 1
        str_log = str(a) + ',' + str(step) + ',' + str(self.all_time[client_id])  + ',' + "desparsify_time"+ ',' + str(
                self.waiting_time[client_id]) + ',' + str(self.ps_time[client_id])
        self.wait_epoch.append(str_log)
        self.all_time[client_id] = 0
        self.waiting_time[client_id] = 0
        self.ps_time[client_id] = 0
        time5 = time.time() - time1
        return time5, global_para,flag1

    def BLocal(self, args, dic):
        # print('BSP')
        flag1 = True
        time1 = time.time()
        self.flag = True
        # print('dic',dic)
        client_id = args[0] - 1
        step = args[1]
        endflag = args[2]
        self.client_dict[client_id] = step
        self.data_dict[client_id] = dic
        u_list = [u for u in self.client_dict if
                  self.client_dict[u] == step]
        while len(u_list) < self.wait_client_sc and self.finish_node==0:
            time.sleep(0.001)
            u_list = [u for u in self.client_dict if self.client_dict[u] == step]
        if self.finish_node!=0 or not self.continue_flag:
            w_avg = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
            flag1 = False
            time1 = time.time()-time1
            return time1,w_avg,flag1       
        time2 = time.time() - time1
        self.waiting_time[client_id] += time2
        time3 = time.time()
        lock.acquire()
        if self.flag:
            self.flag = False
            w_avg = self.data_dict[0]
            for k in w_avg.keys():
                w_avg[k].mul_(self.part[0])
                for i in range(1, len(self.data_dict)):
                    w_avg[k].add_(self.data_dict[i][k].mul_(self.part[i])) 
                self.global_para[k].add_(w_avg[k])
            self.model.load_state_dict(self.global_para, strict=False)    
            self.global_para = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        lock.release()
        global_para ={k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        time4 = time.time() - time3
        self.ps_time[client_id] += time4
        self.all_time[client_id] += (time.time() - time1)
        a = client_id + 1
        str_log = str(a) + ',' + str(step) + ',' + str(self.all_time[client_id])  + ',' + "desparsify_time"+ ',' + str(
                self.waiting_time[client_id]) + ',' + str(self.ps_time[client_id])
        self.wait_epoch.append(str_log)
        self.all_time[client_id] = 0
        self.waiting_time[client_id] = 0
        self.ps_time[client_id] = 0
        time5 = time.time() - time1
        return time5, global_para,flag1

    def Local(self, args, dic):
        # print('BSP')
        flag1 = True
        time1 = time.time()
        self.flag = True
        # print('dic',dic)
        client_id = args[0] - 1
        step = args[1]
        endflag = args[2]
        self.client_dict[client_id] = step
        self.data_dict[client_id] = dic
        u_list = [u for u in self.client_dict if
                  self.client_dict[u] == step]
        while len(u_list) < self.wait_client_sc and self.finish_node==0:
            time.sleep(0.001)
            u_list = [u for u in self.client_dict if self.client_dict[u] == step]
        if self.finish_node!=0 or not self.continue_flag:
            w_avg = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
            flag1 = False
            time1 = time.time()-time1
            return time1,w_avg,flag1
        time2 = time.time() - time1
        self.waiting_time[client_id] += time2
        time3 = time.time()
        lock.acquire()
        if self.flag:
            self.flag = False
            w_avg = self.data_dict[0]
            for k in w_avg.keys():
                w_avg[k].mul_(self.part[0])
                for i in range(1, len(self.data_dict)):
                    w_avg[k].add_(self.data_dict[i][k].mul_(self.part[i])) 
            self.mean_tensor_list = w_avg
            self.model.load_state_dict(w_avg, strict=False)
        lock.release()
        tensor_list={k: v.cpu() for k, v in self.mean_tensor_list.items()}
        time4 = time.time() - time3
        self.ps_time[client_id] += time4
        self.all_time[client_id] += (time.time() - time1)
        a = client_id + 1
        str_log = str(a) + ',' + str(step) + ',' + str(self.all_time[client_id])  + ',' + "desparsify_time"+ ',' + str(
                self.waiting_time[client_id]) + ',' + str(self.ps_time[client_id])
        self.wait_epoch.append(str_log)
        self.all_time[client_id] = 0
        self.waiting_time[client_id] = 0
        self.ps_time[client_id] = 0
        time5 = time.time() - time1
        return time5, tensor_list,flag1

    def Local1(self, args, dic):
        # print('BSP')
        time1 = time.time()
        self.flag = True
        flag1 = True
        # print('dic',dic)
        client_id = dic['client'] - 1
        step = dic['step']
        endflag = dic['end_flag']
        degradients = []
        for tensors in args:
            values, indices, ctx  = tensors
            afterdesparsify = self.desparsify(values, indices, ctx)
            degradients.append(afterdesparsify)
        gradients = [p.cpu() for p in degradients]
        self.data_dict[client_id] = gradients
        self.client_dict[client_id] = step
        time2 = time.time() - time1
        self.desparsify_time[client_id] += time2
        time3 = time.time()
        u_list = [u for u in self.client_dict if
                  self.client_dict[u] == step]
        while len(u_list) < self.wait_client_sc and self.finish_node==0:
            time.sleep(0.001)
            u_list = [u for u in self.client_dict if self.client_dict[u] == step]
        self.waiting_time[client_id] += (time.time() - time3)
        time4 = time.time()
        if self.finish_node!=0:
            w_avg = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
            flag1 = False
            time1 = time.time()-time1
            return w_avg,flag1,time1
        lock.acquire()
        if self.flag:
            self.flag = False
            for i in range(1,self.wait_client_sc):
                for j in range(len(self.data_dict[0])):
                    self.data_dict[0][j].add_(self.data_dict[i][j])                        
            w_avg = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
            for g, p in zip(self.data_dict[0], w_avg.items()):
                    p[1].add_(g.cpu().mul_(0.1))     
            self.mean_tensor_list = w_avg
            self.model.load_state_dict(self.mean_tensor_list, strict=False)
        lock.release()
        w_avg = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        self.ps_time[client_id] += (time.time() - time4)
        self.all_time[client_id] += (time.time() - time1)
        a = client_id + 1
        str_log = str(a) + ',' + str(step) + ',' + str(self.all_time[client_id]) + ',' + str(self.desparsify_time[client_id]) + ',' + str(
                self.waiting_time[client_id]) + ',' + str(self.ps_time[client_id])
        self.wait_epoch.append(str_log)
        self.all_time[client_id] = 0
        self.waiting_time[client_id] = 0
        self.desparsify_time[client_id] = 0
        self.ps_time[client_id] = 0
        time1 = time.time()-time1
        return w_avg,flag1,time1

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes parameters remotely.
    def get_param_rrefs(self, cid):
        self.waiting_time[cid - 1] = 0
        self.all_time[cid - 1] = 0
        self.desparsify_time[cid - 1] = 0
        self.ps_time[cid - 1] = 0
        if cid == 1:
            self.global_start_time = time.time()
            t = Timer(10.0, self.test)
            t.start()
        param_rrefs = {k: v.cpu() for k, v in self.model.state_dict().items()}
        return param_rrefs

    def test(self): 
        time1 = time.time()
        self.test_num += 1
        lock.acquire()
        param_rrefs = {k: v.cuda() for k, v in self.model.state_dict().items()}
        self.test_model.load_state_dict(param_rrefs)
        lock.release()
        para_time = time.time()-time1
        time2 = time.time()
        test_loss = 0
        train_loss = 0
        correct = 0
        total = 0
        # Use GPU to evaluate if possible
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_model.train()
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.test_model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        #nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
        nowTime = time.time() - self.global_start_time 
        train_time = time.time() - time2
        str_log = str(self.test_num) + ',' + str(nowTime)+ ',' + str(para_time) + ',' + str(train_time) + ',' + str(train_loss) + ',' + str(acc)
        self.monitor1.append(str_log)
        if train_loss < 0.02:
            self.continue_num += 1
        else:
            self.continue_num = 0
        if  self.continue_num >=10:
            self.continue_flag =False
        if not self.change_flag and train_loss<0.3:
            self.change_flag=True
        time3 = time.time()
        correct = 0
        total = 0
        self.test_model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.test_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        test_loss = test_loss / len(test_loader)
        #nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
        nowTime = time.time() - self.global_start_time
        test_time = time.time() - time3
        str_log = str(self.test_num) + ',' + str(nowTime) + ',' + str(test_time) + ',' + str(test_loss) + ',' + str(acc)
        self.monitor2.append(str_log)
        print("server",str_log)
        time4 =time.time() - time1
        if time4 > 30.0:
            time_val=0.01
        else:
            time_val=30.0 - time4
        if self.finish_node < self.wait_client_sc and self.continue_flag:
            t = Timer(time_val, self.test)
            t.start()

    def finish(self):
        self.finish_node += 1
        if self.finish_node == self.wait_client_sc:
            f = './log/' + str(train_type) +  "_" + str(learning_rate) + "_" + "sever.csv"
            with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                file.write('client_id' + ',' + 'all_time' + ',' + 'waiting_time' + ',' + 'ps_time' +  "\n")
                for key in self.wait_epoch:
                    file.write(str(key) + "\n")
                file.close()
            nowTime = datetime.datetime.now().strftime('%m-%d')  # 现在
            f = './log/' + str(train_type) + "_" + str(learning_rate) + "_" + str(nowTime) + ".csv"
            with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                file.write('client_id' + ',' + 'time' + ',' + 'state' + "\n")
                for key in self.monitor:
                    file.write(str(key) + "\n")
                file.close()
            f = './log/' + str(train_type) + "_" + str(learning_rate) + "_" + "train_sever" + ".csv"
            with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                file.write('test_num' + ',' + 'nowTime' + ',' + 'para_time'+ ',' + 'train_time' + ',' + 'train_loss' + ',' + 'acc' + "\n")
                for key in self.monitor1:
                    file.write(str(key) + "\n")
                file.close()
            f = './log/' + str(train_type) + "_" + str(learning_rate) + "_" + "test_sever" + ".csv"
            with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                file.write('test_num' + ',' + 'nowTime' + ',' + 'test_time' + ',' + 'test_loss' + ',' + 'acc' + "\n")
                for key in self.monitor2:
                    file.write(str(key) + "\n")
                file.close()
        return True



param_server = None
global_lock = Lock()


def get_parameter_server():
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer()
        return param_server


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size,
                 rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=60000))
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")
    
# --------- Trainers --------------------

class Partition(object):  # 切数据

    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):  # 切数据，调用第一个
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(args.random_seed)  # 设置随机数生成种子，同样的种子会生成同样的随机数。
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]  # 生成标记data索引的一个列表
        rng.shuffle(indexes)  # 使用种子随机排列列表
        for frac in sizes:
            part_len = int(frac * data_len)  # 按size取数据长
            self.partitions.append(indexes[0:part_len])  # 向partitions列表中加入索引列表的前数据长那么多的数据索引
            indexes = indexes[part_len:]  # 将索引列表已取得的值删除

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(partition_sizes):  # 数据分割
    """ Partitioning MNIST """
    dataset = datasets.CIFAR10('../CIFAR10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                               ]))
    size = world_size - 1
    # partition_sizes = [1.0 / size for _ in range(size)]  # 每个机器占多少分之一
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(args.rank - 1)
    train_set = torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True ,num_workers=data_worker)  # 分割的数据加载成纯set
    print(partition_sizes)
    # write_log(partition_sizes,rank)
    return train_set


# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
dic = {}
epoch_time = 0
iter_time = 0
comm_time = 0
train_time = 0
test_time = 0
sparsify_time = 0
monitor = []
test1 = []
list1 = []
time_ma = []
diff_para = {}
glob_para = {}
start_time = 0
time_monitor = []

class Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VGG('VGG16').cuda()
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        warm_up_with_cosine_lr = lambda epoch: epoch / warmup_epoch if epoch <= warmup_epoch else 0.5 * (
                math.cos((epoch - warmup_epoch) / (max_epoch - warmup_epoch) * math.pi) + 1)
        warm_up_with_step_lr = lambda epoch: epoch / warmup_epoch if epoch <= warmup_epoch else 1.0*( 0.5**((epoch - warmup_epoch)//30) )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_cosine_lr)
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server)
        self.H = 2
        self.h = []
        self.global_para={k: v.cuda() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        self.pre_num = 0
        # self.compress_ratio = 0.3
        # self.sample_ratio = 0.6
        self.attributes ={}
        self.compress_ratio = compress_ratio
        self.sample_ratio = sample_ratio
        self.compress_upper_bound = 1.3
        self.compress_lower_bound = 0.8
        self.max_adaptation_iters = 10
        self.resample = True
        self.flag = True
        self.change_flag =False
        self.ps_time = 0
        self.compress_time=0
        self.monitor = []
        self.monitor1 = []
        self.monitor2 = []
        self.monitor3 = []
        self.continue_flag = True

    def get_global_param_rrefs(self, cid):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref,
            cid)
        self.global_para = {k: v.cuda() for k, v in remote_params.items() if 'weight' in k or 'bias' in k}
        self.model.load_state_dict(self.global_para,False)

    def wait_finish(self):
        remote_params = remote_method(
            ParameterServer.finish,
            self.param_server_rref,
        )

    def sync(self):
        # tensor_list= [param.data.cpu() for param in self.model.parameters()]
        arg = []
        arg.append(dic['client'])
        arg.append(dic['step'])
        arg.append(dic['end_flag'])
        tensor_list = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        self.ps_time, remote_params ,self.flag= remote_method(
            ParameterServer.Local,
            self.param_server_rref,
            arg,
            tensor_list
        )
        para = {k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(para, False)
    
    def sync2(self):
        # tensor_list= [param.data.cpu() for param in self.model.parameters()]
        arg = []
        arg.append(dic['client'])
        arg.append(dic['step1'])
        arg.append(dic['step2'])
        arg.append(dic['end_flag'])
        new_para={k: v.cuda() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        up_para = {k:torch.sub(new_para[k] , self.global_para[k]).cpu() for k,_ in new_para.items()}
        self.ps_time, remote_params ,self.flag,self.change_flag= remote_method(
            ParameterServer.ALocal,
            self.param_server_rref,
            arg,
            up_para
        )
        self.compress_time= 0
        if dic['step2']==0 and self.change_flag:
            print("optimizer.param_groups",self.optimizer.param_groups[0]['lr'])
            cosine_lr = lambda epoch:  0.5 * (math.cos(epoch /max_epoch  * math.pi) + 1)
            self.optimizer.param_groups[0]['lr']=learning_rate2
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=cosine_lr)
        self.global_para = {k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(self.global_para, False)

    def sync1(self):
        # tensor_list= [param.data.cpu() for param in self.model.parameters()]
        time0=time.time()
        new_para={k: v.cuda() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        up_para = {k:torch.sub(new_para[k] , self.global_para[k]) for k,_ in new_para.items()}
        tensors =[]
        # count = 0
        #提取梯度tensors=[]
        for name, p in up_para.items():
            if 'weight' in name or 'bias' in name:
                tensor = p.data
            else:
                continue
            numel = tensor.numel()  
                #tensor.size()等价于tensor.shape()，形状
            shape = list(tensor.size())
            num_selects = int(math.ceil(numel * compress_ratio))
            #数据打包
            tensor = tensor.view(-1)
            #绝对值
            importance = tensor.abs()
            #？
            samples = importance
            #从samples中确定topK的最小值
            threshold = torch.min(torch.topk(samples, num_selects, 0, largest=True, sorted=False)[0])
            #选出大于等于threshold的值
            mask = torch.ge(importance, threshold)
            #非零元素的索引
            indices = mask.nonzero().view(-1)     
            indices = indices[:num_selects]
            values = tensor[indices]
            indices = indices.float()
            ctx = numel, shape, name
            newtensor = values.cpu(), indices.cpu(), ctx
            tensors.append(newtensor)
        self.compress_time= time.time()-time0
        self.ps_time, remote_params ,self.flag ,self.change_flag= remote_method(
            ParameterServer.SLocal,
            self.param_server_rref,
            tensors,
            dic
        )
        self.global_para = {k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(self.global_para, False)

    def pre_sync(self):
        global iter_time
        global start_time
        time0=time.time()
        if dic['step'] > dic['end_flag']:
            if start_time==0:
                start_time = time.time()
            else:
                self.monitor.append(time.time()-start_time-self.ps_time)
                start_time = time.time()
            self.pre_num+=1
        arg = []
        arg.append(dic['client'])
        arg.append(dic['step'])
        arg.append(time0)
        tensor_list = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        time1,remote_params = remote_method(
            ParameterServer.pre_train,
            self.param_server_rref,
            arg,
            tensor_list
        )
        self.global_para = {k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(self.global_para,False)
        if dic['step'] > dic['end_flag']:
            self.monitor1.append(time1[0])
            self.monitor2.append(time.time() - time1[1])
        self.ps_time=time.time()-time0

    def pre_sync1(self):
        global iter_time
        global start_time
        time0=time.time()
        new_para={k: v.cuda() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        up_para = {k:torch.sub(new_para[k] , self.global_para[k]) for k,_ in new_para.items()}
        tensors =[]
        # count = 0
        #提取梯度tensors=[]
        for name, p in up_para.items():
            if 'weight' in name or 'bias' in name:
                tensor = p.data
            else:
                continue
            numel = tensor.numel()  
                #tensor.size()等价于tensor.shape()，形状
            shape = list(tensor.size())
            num_selects = int(math.ceil(numel * compress_ratio))
            #数据打包
            tensor = tensor.view(-1)
            #绝对值
            importance = tensor.abs()
            #？
            samples = importance
            #从samples中确定topK的最小值
            threshold = torch.min(torch.topk(samples, num_selects, 0, largest=True, sorted=False)[0])
            #选出大于等于threshold的值
            mask = torch.ge(importance, threshold)
            #非零元素的索引
            indices = mask.nonzero().view(-1)     
            indices = indices[:num_selects]
            values = tensor[indices]
            indices = indices.float()
            ctx = numel, shape, name
            newtensor = values.cpu(), indices.cpu(), ctx
            tensors.append(newtensor)
        time_gc= time.time()-time0
        if dic['step'] > dic['end_flag']:
            self.monitor3.append(time_gc)
        time0=time.time()
        if dic['step'] > dic['end_flag']:
            if start_time==0:
                start_time = time.time()
            else:
                self.monitor.append(time.time()-start_time-self.ps_time-time_gc)
                start_time = time.time()
            self.pre_num+=1
        dic['time']=time0
        time1,remote_params = remote_method(
            ParameterServer.pre_train1,
            self.param_server_rref,
            tensors,
            dic
        )
        self.global_para = {k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(self.global_para,False)
        if dic['step'] > dic['end_flag']:
            self.monitor1.append(time1[0])
            self.monitor2.append(time.time() - time1[1])
        self.ps_time=time.time()-time0 

    def get_part(self):
        global iter_time
        for i in range(20):
            self.monitor.remove(max(self.monitor))
            self.monitor1.remove(max(self.monitor1))
            self.monitor2.remove(max(self.monitor2))
            self.monitor3.remove(max(self.monitor3))
            self.monitor.remove(min(self.monitor))
            self.monitor1.remove(min(self.monitor1))
            self.monitor2.remove(min(self.monitor2))
            self.monitor3.remove(min(self.monitor3))
        # time_ma[0] = iter_time 
        time_list = [dic['client'], sum(self.monitor)/len(self.monitor), sum(self.monitor1)*8/len(self.monitor1), sum(self.monitor2)*8/len(self.monitor2), sum(self.monitor3)/len(self.monitor3)]
        h, part,cplist= remote_method(
            ParameterServer.get_part,
            self.param_server_rref,
            time_list,
        )
        print(h, part)
        self.h = h
        self.H = self.h[1][dic['client']-1]
        self.compress_ratio=cplist[dic['client']-1]
        return part
    
    def sparsify(self, tensor, name):
        #改变一个 tensor 的大小或者形状,view()返回的数据和传入的tensor一样，只是形状不同。
        #tensor.view(-1)是生成一维的，只有一行数据
        tensor = tensor.view(-1)
        numel, shape, num_selects, num_samples, top_k_samples = self.attributes[name]
        #绝对值
        importance = tensor.abs()
        #？
        if numel == num_samples:
            samples = importance
        else:
            #torch.randint（）用于生成一个指定范围内的整数
            samples = importance[torch.randint(0, numel, (num_samples,), device=tensor.device)]
        #从samples中确定topK的最小值
        threshold = torch.min(torch.topk(samples, top_k_samples, 0, largest=True, sorted=False)[0])
        #选出大于等于threshold的值
        mask = torch.ge(importance, threshold)
        #非零元素的索引
        indices = mask.nonzero().view(-1)
        #h获取张量元素个数
        num_indices = indices.numel()
        
        #抽样基础上获得的threshold在全部数据上筛选后的元素数量不一定满足num_selects，多的减少，少的增加
        if numel > num_samples:
            for _ in range(self.max_adaptation_iters):
                #print("num_indices",num_indices,"num_selects",num_selects)
                if num_indices > num_selects:
                    if num_indices > num_selects * self.compress_upper_bound:
                        if self.resample:
                            indices = indices[
                                torch.topk(importance[indices], num_selects,
                                           0, largest=True, sorted=False)[1]
                            ]
                            break
                        else:
                            threshold = threshold * self.compress_upper_bound
                    else:
                        break
                elif num_indices < self.compress_lower_bound * num_selects:
                    threshold = threshold * self.compress_lower_bound
                else:
                    break
                mask = torch.ge(importance, threshold)
                indices = mask.nonzero().view(-1)
                num_indices = indices.numel()
        
        indices = indices[:num_selects]
        values = tensor[indices]
        indices = indices.float()
        ctx = numel, shape, name
        return values.cpu(), indices.cpu(), ctx

criterion = nn.CrossEntropyLoss()


def run_training_loop(rank, n_epochs, train_loader, test_loader):
    # Runs the typical neural network forward + backward + optimizer step, but
    # in a distributed fashion.
    # Build DistributedOptimizer.
    global epoch_time
    global iter_time
    global comm_time
    global train_time
    global sparsify_time
    time_ma.append(iter_time)
    time_ma.append(comm_time)
    time_ma.append(epoch_time)
    time_ma.append(train_time)
    net = Trainer()
    dic['client'] = rank
    # net.get_global_param_rrefs(rank)
    # str_log = 'epoch' + ',' + 'nowTime' + ',' +'train_time:'+ ','+'total compute time:'  +','+ 'total comm time'+','+\
    #            'train_loss:' +',' + 'Accuracy:'
    # write_log(str_log, rank)
    end_flag = len(train_loader)
    dic['end_flag'] = end_flag
    i = 0
    for epoch in range(0, 10):
        net.model.train()
        # iter_time = 0
        comm_time = 0
        train_time = 0
        train_loss = 0
        correct = 0
        total = 0
        time1 = time.time()
        for data, target in train_loader:
            time2 = time.time()
            data, target = data.cuda(), target.cuda()
            i = i + 1
            dic['step'] = i
            net.optimizer.zero_grad()
            model_output = net.model(data)
            loss = criterion(model_output, target)
            loss.backward()
            net.optimizer.step()
            train_loss += loss.item()
            _, predicted = model_output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            time.sleep((time.time() - time2)*3)
            # iter_time += (time.time() - time2)
            time3 = time.time()
            net.pre_sync1()
            comm_time += (time.time() - time3)
        # net.scheduler.step()
        train_time += (time.time() - time1)
        # iter_time += (train_time -comm_time)
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
        str_log = str(epoch) + ',' + str(nowTime) + ',' + str(train_time) + ',' +str(epoch_time) + ',' + str(iter_time) + \
                  ',' + str(comm_time) +  ',' + str(sparsify_time) +',' + str(train_loss / len(train_loader)) + ',' + str(100. * correct / total)
        monitor.append(str_log)
    for i in range(len(net.monitor)):
        str_log = str(net.monitor[i]) + ',' + str(net.monitor1[i]) + ',' + str(net.monitor2[i])
        time_monitor.append(str_log)
    write1(rank)
    train_loader = partition_dataset(net.get_part())
    i = 0
    i1 = 0
    i2 = 0
    dic['step1'] = i1
    dic['step2'] = i2
    net.get_global_param_rrefs(rank)
    k=int(epoch2/200)
    while(net.flag and i2<epoch2):
        net.model.train()
        iter_time = 0
        comm_time = 0
        epoch_time = 0
        train_time = 0
        train_loss = 0
        sparsify_time = 0
        correct = 0
        total = 0
        if  not net.flag:
            break  
        time1 = time.time()
        for data, target in train_loader:
            time2 = time.time()
            data, target = data.cuda(), target.cuda()
            i = i + 1
            net.optimizer.zero_grad()
            model_output = net.model(data)
            loss = criterion(model_output, target)
            loss.backward()
            net.optimizer.step()
            train_loss += loss.item()
            _, predicted = model_output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if not net.change_flag:
                net.H = net.h[1][dic['client']-1]
            else:
                net.H = net.h[0][dic['client']-1] 
            time.sleep((time.time() - time2)*3)
            iter_time += (time.time() - time2)
            if i % net.H == 0 and net.flag and not net.change_flag:
                i1+=1
                dic['step1'] = i1
                time3 = time.time()
                net.sync2()
                if i1%6==0:
                    net.scheduler.step()
                time.sleep((time.time() - time3 -net.ps_time-net.compress_time)*7)
                comm_time += (time.time() - time3 -net.ps_time-net.compress_time)
                sparsify_time+=net.compress_time
            elif i % net.H == 0 and net.flag and net.change_flag: 
                i2+=1
                dic['step2'] = i2
                time3 = time.time()
                net.sync1()
                time.sleep((time.time() - time3 -net.ps_time-net.compress_time)*7)
                comm_time += (time.time() - time3 -net.ps_time-net.compress_time)
                sparsify_time+=net.compress_time
                if  i2%k==0:
                    net.scheduler.step()
            if  not net.flag:
                break
            epoch_time += (time.time() - time2)
        # time3 = time.time()
        # net.sync()
        # comm_time += (time.time() - time3)
        train_time += (time.time() - time1)
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
        str_log = str(epoch) + ',' + str(nowTime) + ',' + str(train_time) + ',' +str(epoch_time) + ',' + str(iter_time) + \
                  ',' + str(comm_time) +  ',' + str(sparsify_time) +',' + str(train_loss / len(train_loader)) + ',' + str(100. * correct / total)
        test1.append(str_log)
        #get_accuracy(test_loader, net.model, rank, epoch)
    f = './log/' + str(train_type) + "_" + "test"+ "_" + str(epoch2) + "_"+ str(compress_ratio)+ "_" + str(learning_rate2) + "_" + str(rank) + ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write('epoch' + ',' + 'nowTime' + ',' + 'test_time' + ',' + 'test_loss' + ',' + 'acc' + "\n")
        for key in monitor:
            file.write(str(key) + "\n")
        file.close()
    f = './log/' + str(train_type) + "_" + "train" + "_"+ str(epoch2) + "_"+ str(compress_ratio)+ "_" + str(learning_rate2) + "_" + str(rank) + ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(
            'epoch' + ',' + 'nowTime' + ',' + 'train_time'+',' + 'epoch_time' + ',' + 'iter_time' + ',' + 'comm_time' + 'sparsify_time' + ','+ ',' + 'train_loss' + ',' + 'acc' + "\n")
        for key in test1:
            file.write(str(key) + "\n")
        file.close()
    # get_accuracy(test_loader, net.model, rank, n_epochs)
    net.wait_finish()

def write1(client_id):  # 写日志函数
    f = './log/' + str(client_id) + "_" + "time_test"+ ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write('cTime' + ',' + 'pushTime' + ',' + 'pull_time' +"\n")
        for key in time_monitor:
            file.write(str(key) + "\n")
        file.close()
        
def get_accuracy(test_loader, model, rank, epoch):
    time1 = time.time()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    # Use GPU to evaluate if possible
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    test_loss = test_loss / len(test_loader)
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    test_time = time.time() - time1
    str_log = str(epoch) + ',' + str(nowTime) + ',' + str(test_time) + ',' + str(test_loss) + ',' + str(acc)
    monitor.append(str_log)
    print(str_log)


# Main loop for trainers.

def run_worker(rank, world_size, n_epochs, train_loader, test_loader):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=6000)
    )

    print(f"Worker {rank} done initializing RPC")

    run_training_loop(rank, n_epochs, train_loader, test_loader)
    rpc.shutdown()


# --------- Launcher --------------------


if __name__ == '__main__':

    assert args.rank is not None, "must provide rank argument."
    # assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['GLOO_SOCKET_IFNAME']='eth0'
    os.environ['TP_SOCKET_IFNAME']='eth0'
    os.environ['TENSORPIPE_SOCKET_IFNAME']='eth0'
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    # os.environ['CUDA_VISIBLE_DEVICE'] = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    processes = []
    seed_torch(args.random_seed)
    print("train_type:", train_type)
    if args.rank == 0:
        print("device", args.device)
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        # Get data to train on
        print("device", args.device)
        partition_sizes = [1.0 / (world_size-1) for _ in range(world_size-1)]  # 每个机器占多少分之一
        train_loader = partition_dataset(partition_sizes)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(
            root='../CIFAR10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False)

        # start training worker on this node
        p = mp.Process(
            target=run_worker,
            args=(
                args.rank,
                world_size, args.n_epochs,
                train_loader,
                test_loader))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
