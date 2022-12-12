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
from threading import Timer
from sympy import Symbol, solve
import copy

lock = Lock()
lock1 = Lock()
lock2 = Lock()
parser = argparse.ArgumentParser(
    description="Parameter-Server RPC based training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="""Total number of participating processes. Should be the sum of
       master node and all training nodes.""")
parser.add_argument("--train_type", type=str, default='ADSP')
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--time", type=float, default=60.0)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=42)
parser.add_argument("--warmup_epoch", type=int, default=5)
parser.add_argument("--T", type=int, default=20)
parser.add_argument("--data_worker", type=int, default=1)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=100,
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
world_size = args.world_size
time_T = args.time
st = args.T
rank = args.rank
train_type = args.train_type
learning_rate = args.learning_rate
batch_size = args.batch_size
warmup_epoch = args.warmup_epoch
max_epoch = args.n_epochs
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
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_cosine_lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size =50, gamma = 0.1, last_epoch=-1)
        self.waiting_time = {}
        self.wait_epoch = []
        self.monitor = []
        self.monitor1 = []
        self.monitor2 = []
        self.new_epoch = 0
        self.mean_tensor_list = {}
        self.ModelProcessing = None
        self.wait_client_sc = world_size - 1
        self.wait_c = world_size - 1
        self.client_dict = [0 for i in range(self.wait_client_sc)]
        self.finish_node = 0
        self.curr_update_size = 0
        self.sync_layer_list = []
        self.flag = True
        self.client_c = {}
        self.client_r = {}
        self.num_c = {}
        self.num_r = {}
        self.c = 1
        self.r = 0
        self.reward = True
        self.global_start_time=0
        self.test_num=0
        self.continue_flag = True
        self.continue_num = 0

    def get_c(self, arg):
        time1 = time.time()
        self.flag = 0
        client_id = arg[0]-1
        a = arg[1]
        b = arg[2]
        self.client_c[client_id] = a
        self.num_c[client_id] = b
        str_log=str(client_id)+','+'get_c in'
        self.monitor.append(str_log)
        u_list = [u for u in self.num_c if self.num_c[u] == b]
        time1=time.time()
        while len(u_list) < self.wait_client_sc and self.finish_node==0:
            time.sleep(0.01)
            u_list = [u for u in self.num_c if self.num_c[u] == b]
        if self.finish_node>0:
            str_log=str(client_id)+','+'get_c False'
            self.monitor.append(str_log)
            time2 = time.time() - time1
            return a+1,time2
        lock1.acquire()
        if self.flag == 0:
            self.flag = 1
            max_c = [value for kay, value in self.client_c.items()]
            self.c = max(max_c) + 1
        lock1.release()
        str_log=str(client_id)+','+'get_c out'
        self.monitor.append(str_log)
        time2 = time.time() - time1
        return self.c,time2

    def reward(self, arg):
        time1 = time.time()
        self.reward = True
        self.flag = 0
        client_id = arg[0]-1
        a = arg[1]
        b = arg[2]
        self.client_r[client_id] = a
        self.num_r[client_id] = b
        str_log=str(client_id)+','+'reward in'
        self.monitor.append(str_log)
        u_list = [u for u in self.num_r if self.num_r[u] == b]
        time1=time.time()
        while len(u_list) < self.wait_client_sc and self.finish_node ==0:
            time.sleep(0.01)
            u_list = [u for u in self.num_r if self.num_r[u] == b]
        if self.finish_node>0:
            str_log=str(client_id)+','+'reward False'
            self.monitor.append(str_log)
            time2 = time.time() - time1
            return False,time2
        lock1.acquire()
        if self.flag == 0:
            self.flag = 1
            #self.scheduler.step()
            max_c = [value for kay, value in self.client_r.items()]
            mean = sum(max_c) / len(max_c)
            if mean > self.r:
                self.r = mean
            else:
                self.reward = False
                self.r = 0
        lock1.release()
        str_log=str(client_id)+','+'reward out'
        self.monitor.append(str_log)
        time2 = time.time() - time1
        return self.reward,time2

    def ASP(self, args, dic):
        time1 = time.time()
        client_id = dic['client'] - 1
        step = dic['step']
        endflag = dic['end_flag']
        self.client_dict[client_id] = step
        lock2.acquire()
        self.curr_update_size += 1
        self.optimizer.zero_grad()
        for g, p in zip(args, self.model.named_parameters()):
            if g is not None:
                p[1].grad = g.cpu()
        self.optimizer.step()
        mean_tensor_list = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        lock2.release()
        time2 = time.time() - time1
        return mean_tensor_list,time2,self.continue_flag

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes parameters remotely.
    def get_param_rrefs(self, cid):
        self.waiting_time[cid - 1] = 0
        param_rrefs = {k: v.cpu() for k, v in self.model.state_dict().items()}
        if cid == 1:
            self.global_start_time = time.time()
            t = Timer(10.0, self.test)
            t.start()
        return param_rrefs

    def finish(self):
        self.finish_node += 1
        if self.finish_node == self.wait_client_sc:
            f = './log/' + str(train_type) + "_" + "sever.csv"
            with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                file.write('client_id' + ',' + 'all_time' + ',' + 'waiting_time' + ',' + 'ps_time' +  "\n")
                for key in self.wait_epoch:
                    file.write(str(key) + "\n")
                file.close()
            nowTime = datetime.datetime.now().strftime('%m-%d')  # 现在
            f = './log/' + str(train_type) + "_" + str(nowTime) + ".csv"
            with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                file.write('client_id' + ',' + 'time' + ',' + 'state' + "\n")
                for key in self.monitor:
                    file.write(str(key) + "\n")
                file.close()
            f = './log/' + str(train_type) + "_" + "train_sever" + ".csv"
            with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                file.write('test_num' + ',' + 'nowTime' + ',' + 'para_time'+ ',' + 'train_time' + ',' + 'train_loss' + ',' + 'acc' + "\n")
                for key in self.monitor1:
                    file.write(str(key) + "\n")
                file.close()
            f = './log/' + str(train_type) + "_" + "test_sever" + ".csv"
            with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                file.write('test_num' + ',' + 'nowTime' + ',' + 'test_time' + ',' + 'test_loss' + ',' + 'acc' + "\n")
                for key in self.monitor2:
                    file.write(str(key) + "\n")
                file.close()
        return True

    def test(self):
        time1 = time.time()
        self.test_num += 1
        lock2.acquire()
        param_rrefs = {k: v.cuda() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        self.test_model.load_state_dict(param_rrefs,False)
        lock2.release()
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
        if train_loss < 0.05:
            self.continue_num += 1
        else:
            self.continue_num = 0
        if  self.continue_num >=10:
            self.continue_flag =False
        nowTime = time.time() - self.global_start_time 
        train_time = time.time() - time2
        str_log = str(self.test_num) + ',' + str(nowTime)+ ',' + str(para_time) + ',' + str(train_time) + ',' + str(train_loss) + ',' + str(acc)
        self.monitor1.append(str_log)
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
        if self.finish_node < self.wait_client_sc and self.continue_flag == True:
            t = Timer(time_val, self.test)
            t.start()


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


def partition_dataset():  # 数据分割
    """ Partitioning MNIST """
    dataset = datasets.CIFAR10('../CIFAR10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                               ]))
    size = world_size - 1
    partition_sizes = [1.0 / size for _ in range(size)]  # 每个机器占多少分之一
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
lt = []
iter_time = 0
comm_time = 0
train_time = 0
epoch_time = 0
test_time = 0
monitor = []
test1 = []
test2 = []
lock1 = Lock()
lock2 = Lock()

class Trainer(nn.Module):
    def __init__(self,train_loader):
        super().__init__()
        self.model = VGG('VGG16').cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.dataloader = train_loader
        self.data_iterator = iter(train_loader)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        warm_up_with_cosine_lr = lambda epoch: epoch / warmup_epoch if epoch <= warmup_epoch else 0.5 * (
                math.cos((epoch - warmup_epoch) / (max_epoch - warmup_epoch) * math.pi) + 1)
        cosine_lr = lambda epoch: 0.5 * ( math.cos(epoch/max_epoch * math.pi) + 1)
        warm_up_with_step_lr = lambda epoch: epoch / warmup_epoch if epoch <= warmup_epoch else 1.0*( 0.5**((epoch - warmup_epoch)//30) )
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_cosine_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size =50, gamma = 0.5, last_epoch=-1)
        self.iter_time = 0
        self.comm_time = 0
        self.train_time = 0
        self.epoch_time = 0
        self.pstime = 0
        self.r = 0  # reward
        self.g_c = 0  # 全局次数
        self.l_c = 0  # 本机提交次数
        self.c = 0  # 时间间隔内提交次数
        self.m_c = 0  # 时间间隔内提交次数计数器
        self.T1 = 0  # 单次通信的时间
        self.T = time_T  # T大小
        self.T_c = st  # search周期10个self.T
        self.E_T = 0  # 计数器
        self.end_flag = 0
        self.next_flag = True
        self.r_continue = True
        self.loss_sum = 0#loss
        self.r_num = 0#reward次数
        self.c_num = 0#getc次数
        self.s_num = 0#search次数
        self.lock = Lock()
        self.global_para={}
        self.sum_grad =[]
        self.lt=[]
        self.T1_list=[]
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server)

    def get_global_param_rrefs(self, cid):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref,
            cid)
        para = {k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(para)

    def wait_finish(self):
        lock.acquire()
        remote_params = remote_method(
            ParameterServer.finish,
            self.param_server_rref,
        )
        lock.release()

    def get_c(self):
        time1 = time.time()
        self.c_num += 1
        log_str="get_c in" + ','+str(self.c_num)
        # write_log3(log_str, rank)
        ci = [rank, self.l_c, self.c_num]
        remote_params ,pstime = remote_method(
            ParameterServer.get_c,
            self.param_server_rref,
            ci
        )
        time2 = time.time()
        # comm_time = time2 - time1 - pstime
        # time.sleep(comm_time)
        self.g_c = remote_params
        self.m_c = self.g_c - self.l_c
        str_log="reward out"+','+str(self.c)
        # write_log3(str_log, rank) 

    def reward(self):
        #test2.append("reward in")
        #write_log3("reward in", rank)
        self.task()
        time1 = time.time()
        self.r_num += 1
        log_str="reward in" + ','+str(self.r_num)
        # write_log3(log_str, rank)
        #print(rank,"reward in",self.r_num)
        next = self.T/(self.m_c+1) - self.T1>0
        ci = [rank, self.r,self.r_num,next]
        self.r_continue ,flag1,pstime= remote_method(
            ParameterServer.reward,
            self.param_server_rref,
            ci
        )
        if self.r_continue and flag1 :
            self.g_c = self.g_c+1
            self.m_c = self.m_c+1
        elif self.r_continue and not flag1 :
            self.r_continue = False
        else:
            self.g_c = self.g_c-1
            self.m_c = self.m_c-1
        # time2 = time.time()
        # comm_time = time2 - time1 - pstime
        # time.sleep(comm_time)
        # test2.append("reward out")
        print(rank,self.m_c)
        str_log="reward out"+','+str(self.r_continue)
        # write_log3(str_log, rank)


    def sync(self):
        # tensor_list= [param.data.cpu() for param in self.model.parameters()]
        #print(rank,"sync in")
        #test2.append("sync in")
        grads=[]
        for p in self.sum_grad:
            grad = None if p is None else p.cpu()
            grads.append(grad)
        # write_log3("lock1 out", rank)
        #write_log3("remote in", rank)
        remote_params ,self.pstime,flag= remote_method(
            ParameterServer.ASP,
            self.param_server_rref,
            grads,
            dic
        )
        self.next_flag = flag
        # write_log3("sync out", rank)
        self.global_para={k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(self.global_para, strict=False) 
        #test2.append("sync out1")
        #print(rank,"sync out")

    def train(self):
        time1=time.time()
        for i in range(self.m_c):
            time2=time.time()
            if self.T/self.m_c - self.T1<0:
                print("出错了",self.m_c,self.T/self.m_c,self.T1)
            while(time.time()-time2)<(self.T/self.m_c - self.T1):
                time3=time.time()
                self.model.train()
                lr=self.optimizer.param_groups[0]['lr']
                try:
                    data, target = next(self.data_iterator)
                    data, target = data.cuda(), target.cuda()
                except StopIteration:  # When the epoch ends, start a new epoch.
                    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
                    str_log = str(nowTime) + ','  + str(self.iter_time) + ',' + str(self.epoch_time) + \
                  ',' + str(self.comm_time) + ',' + str(self.loss_sum / len(self.dataloader)) 
                    write_log2(str_log,rank)
                    self.loss_sum=0
                    self.comm_time=0
                    self.iter_time=0
                    timee=time.time()
                    self.data_iterator = iter(self.dataloader)
                    self.epoch_time=time.time()-timee
                    data, target = next(self.data_iterator)
                    data, target = data.cuda(), target.cuda()
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.loss_sum += loss.item()
                self.lt.append(loss)
                loss.backward()
                k = 0
                for p in self.model.parameters():
                    self.sum_grad[k] = None if p.grad is None else self.sum_grad[k].add_(p.grad.cuda().mul_(lr))
                    k += 1 
                time.sleep((time.time() - time3)*3)
                self.iter_time+=(time.time() - time3)
            time3=time.time()
            self.sync()
            self.l_c +=1
            j = len(self.sum_grad)
            for i in range(0, j):
                self.sum_grad[i] = None if self.sum_grad[i] is None else torch.zeros_like(self.sum_grad[i])
            time.sleep((time.time() - time3-self.pstime)*7)
            self.comm_time+=(time.time() - time3-self.pstime)
            if len(self.T1_list)<10:
                self.T1_list.append( time.time()-time3)
                self.T1= sum(self.T1_list)/len(self.T1_list)
            else:
                self.T1_list[self.l_c%10]=time.time()-time3
            self.T1= sum(self.T1_list)/len(self.T1_list)  

    def task(self):
        if len(self.lt)>=3:
            x = Symbol('x')
            y = Symbol('y')
            z = Symbol('z')
            l1 = self.lt[0]
            l2 = self.lt[int(len(self.lt) / 2)]
            l3 = self.lt[len(self.lt) - 1]
            solved_value = solve([1.0 / y + z - l1, 1.0 / (0.5 * x ** 2 + y) + z - l2, 1.0 / (x ** 2 + y) + z - l3],
                                 [x, y, z])
        # print("solved_value",solved_value)
            if solved_value!=[]:
                self.r = solved_value[0][0] ** 2 / ((1.0 / (l3 - solved_value[0][2])) - solved_value[0][1])
            self.lt.clear()


criterion = nn.CrossEntropyLoss()

def run_training_loop(rank, n_epochs, train_loader, test_loader):
    # Runs the typical neural network forward + backward + optimizer step, but
    # in a distributed fashion.
    # Build DistributedOptimizer.
    global iter_time
    global comm_time
    global train_time
    global epoch_time
    global lt
    global sum_grad
    net = Trainer(train_loader)
    dic['client'] = rank
    net.get_global_param_rrefs(rank)
    str_log = 'epoch' + ',' + 'nowTime' + ',' + 'test_time' + ',' + 'test_loss' + ',' + 'acc'
    write_log1(str_log, rank)
    # str_log = 'epoch' + ',' + 'nowTime' + ',' + 'train_time' + ',' + 'iter_time' + ',' +'epoch_time' + ','+ 'comm_time' + ',' + 'train_loss' + ',' + 'acc'
    str_log = 'nowTime' + ',' + 'iter_time' + ',' +'epoch_time' + ','+ 'comm_time' + ',' + 'train_loss' 
    write_log2(str_log, rank)
    end_flag = len(train_loader)
    dic['end_flag'] = end_flag
    k = 0
    dic['step'] = k
    for p in net.model.parameters():
        grad = None if p.grad is None else p.grad.data.cuda()
        net.sum_grad.append(grad)
    while net.s_num < n_epochs and net.next_flag:
        net.get_c()
        num1=copy.deepcopy(net.T_c)
        net.r_continue = True
        while net.r_continue and num1>0:
            net.lt.clear()
            net.train()
            net.reward()
            num1=num1-1
        for i in range(num1):
            net.train()
        net.s_num += 1
        net.scheduler.step()
    print(rank,"end_flag")
    net.wait_finish()


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
    #monitor.append(str_log)
    write_log1(str_log,rank)

def write_log1(str_log, rank):  # 写日志函数
    f = './log/' + str(train_type) + "_" + "test" + "_" + str(learning_rate) + "_" + str(time_T)  + "_" + str(rank) + ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(str(str_log) + "\n")
        file.close()

def write_log2(str_log, rank):  # 写日志函数
    f =  './log/' + str(train_type) + "_" + "train" + "_" + str(learning_rate) + "_" + str(time_T)  + "_" + str(rank) + ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(str(str_log) + "\n")
        file.close()

def write_log3(str_log, rank):  # 写日志函数
    f = './log/' + str(train_type) + "_" + "monitor" + "_" + str(learning_rate) + "_" + str(time_T)  + "_" + str(rank) + ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(str(str_log) + "\n")
        file.close()
# Main loop for trainers.

def run_worker(rank, world_size, n_epochs, train_loader, test_loader):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=600)
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
        dataset = datasets.CIFAR10('../CIFAR10', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                   ]))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False ,num_workers=data_worker)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = datasets.CIFAR10(
            root='../CIFAR10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=500, shuffle=False)
        print("device", args.device)
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        # Get data to train on
        print("device", args.device)
        train_loader = partition_dataset()
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(
            root='../CIFAR10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False ,num_workers=data_worker)

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