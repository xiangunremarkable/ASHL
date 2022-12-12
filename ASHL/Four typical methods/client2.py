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

lock = Lock()
parser = argparse.ArgumentParser(
    description="Parameter-Server RPC based training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="""Total number of participating processes. Should be the sum of
       master node and all training nodes.""")
parser.add_argument("--train_type", type=str, default='BSP')
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--grad_value", type=float, default=0.5)
parser.add_argument("--H", type=int, default=2)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=0.8)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--warmup_epoch", type=int, default=5)
parser.add_argument("--data_worker", type=int, default=1)
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
world_size = args.world_size
train_type = args.train_type
learning_rate = args.learning_rate
batch_size = args.batch_size
data_worker = args.data_worker
warmup_epoch = args.warmup_epoch
max_epoch = args.n_epochs
device = args.device
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
        self.model = VGG('VGG16').cuda()
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)
        warm_up_with_cosine_lr = lambda epoch: epoch / warmup_epoch if epoch <= warmup_epoch else 0.5 * (
                math.cos((epoch - warmup_epoch) / (max_epoch - warmup_epoch) * math.pi) + 1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_cosine_lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size =50, gamma = 0.1, last_epoch=-1)
        self.waiting_time = {}
        self.ps_time = {}
        self.all_time = {}
        self.wait_epoch = []
        self.monitor = []
        self.new_epoch = 0
        self.mean_tensor_list = {}
        self.ModelProcessing = None
        self.wait_client_sc = world_size - 1
        self.client_dict = {}
        self.finish_node = 0
        self.curr_update_size = 0
        self.sync_layer_list = []
        self.flag = True

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    def BSP(self, args, dic):
        # print('BSP')
        time1 = time.time()
        self.flag = True
        # print('dic',dic)
        client_id = dic['client'] - 1
        step = dic['step']
        endflag = dic['end_flag']
        grads=[]
        for p in args:
            grad = None if p is None else p.cuda()
            grads.append(grad)
        self.data_dict[client_id] = grads
        self.client_dict[client_id] = step
        u_list = [u for u in self.client_dict if
                  self.client_dict[u] == step]
        while len(u_list) < self.wait_client_sc:
            time.sleep(0.01)
            u_list = [u for u in self.client_dict if self.client_dict[u] == step]
        time2 = time.time() - time1
        self.waiting_time[client_id] += time2
        time3 = time.time()
        lock.acquire()
        if self.flag:
            self.flag = False
            mean_list = []
            for tensor_id in range(len(grads)):
                u_tensor_list = [self.data_dict[u][tensor_id] for u in range(len(u_list))]
                mean_list.append(torch.mean(torch.stack(u_tensor_list), 0))
            self.optimizer.zero_grad()
            for g, p in zip(mean_list, self.model.named_parameters()):
                if g is not None:
                    p[1].grad = g.cuda()
            self.mean_tensor_list = {k: v.cuda() for k, v in self.model.state_dict().items() if
                                     'weight' in k or 'bias' in k}
            if step % endflag == 0:
                self.scheduler.step()
        lock.release()
        tensor_list={k: v.cpu() for k, v in self.mean_tensor_list.items()}
        time4 = time.time() - time3
        self.ps_time[client_id] += time4
        self.all_time[client_id] += (time.time() - time1)
        if step % endflag == 0:
            a = client_id + 1
            str_log = str(a) + ',' + str(self.all_time[client_id]) + ',' + str(self.waiting_time[client_id]) + ',' + str(self.ps_time[client_id])
            self.wait_epoch.append(str_log)
            self.all_time[client_id] = 0
            self.waiting_time[client_id] = 0
            self.ps_time[client_id] = 0
        time5 = time.time() - time1
        return time5, tensor_list

    def SSP(self, args, dic):
        time1 = time.time()
        client_id = dic['client'] - 1
        step = dic['step']
        endflag = dic['end_flag']
        self.client_dict[client_id] = step
        self.data_dict[client_id] = args
        grads=[]
        for p in args:
            grad = None if p is None else p.cuda()
            grads.append(grad)
        u_list = [u for u in self.client_dict if
                  self.client_dict[u] > 0]
        findmin = [self.client_dict[u] for u in u_list]
        while (step - 3) > (min(findmin)):
            time.sleep(0.01)
            findmin = [self.client_dict[u] for u in u_list]
        time2 = time.time() - time1
        self.waiting_time[client_id] += time2
        time3 = time.time()
        lock.acquire()
        self.curr_update_size += 1
        self.optimizer.zero_grad()
        for g, p in zip(grads, self.model.named_parameters()):
            if g is not None:
                p[1].grad = g.cuda()
        self.optimizer.step()
        mean_tensor_list = {k: v.cuda() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        if self.curr_update_size % (self.wait_client_sc * endflag) == 0:
            self.scheduler.step()
        lock.release()
        tensor_list={k: v.cpu() for k, v in mean_tensor_list.items()}
        time4 = time.time() - time3
        self.ps_time[client_id] += time4
        self.all_time[client_id] += (time.time() - time1)
        if step % endflag == 0:
            a = client_id + 1
            str_log = str(a) + ',' + str(self.all_time[client_id]) + ',' + str(
                self.waiting_time[client_id]) + ',' + str(self.ps_time[client_id])
            self.wait_epoch.append(str_log)
            self.all_time[client_id] = 0
            self.waiting_time[client_id] = 0
            self.ps_time[client_id] = 0
        time5 = time.time() - time1
        return time5, tensor_list

    def ASP(self, args, dic):
        time1 = time.time()
        client_id = dic['client'] - 1
        step = dic['step']
        endflag = dic['end_flag']
        self.client_dict[client_id] = step
        self.data_dict[client_id] = args
        grads=[]
        for p in args:
            grad = None if p is None else p.cuda()
            grads.append(grad)
        lock.acquire()
        time2 = time.time() - time1
        self.waiting_time[client_id] += time2
        self.curr_update_size += 1
        self.optimizer.zero_grad()
        for g, p in zip(grads, self.model.named_parameters()):
            if g is not None:
                p[1].grad = g.cuda()
        self.optimizer.step()
        mean_tensor_list = {k: v.cuda() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
        #self.mean_tensor_list[client_id] = copy.deepcopy(mean_tensor_list)
        if self.curr_update_size % (self.wait_client_sc * endflag) == 0:
            self.scheduler.step()
        lock.release()
        tensor_list={k: v.cpu() for k, v in mean_tensor_list.items()}
        time3 = time.time() - time1
        self.all_time[client_id] += time3
        if step % endflag == 0:
            a = client_id + 1
            str_log = str(a) + ',' + str(self.all_time[client_id]) + ',' + str(
                self.waiting_time[client_id])
            self.wait_epoch.append(str_log)
            self.all_time[client_id] = 0
            self.waiting_time[client_id] = 0
        time5 = time.time() - time1
        return time5, tensor_list

    def Local(self, args, dic):
        # print('BSP')
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
        while len(u_list) < self.wait_client_sc:
            time.sleep(0.01)
            u_list = [u for u in self.client_dict if self.client_dict[u] == step]
        time2 = time.time() - time1
        self.waiting_time[client_id] += time2
        time3 = time.time()
        lock.acquire()
        if self.flag:
            self.flag = False
            w_avg = self.data_dict[0]
            for k in w_avg.keys():
                for i in range(1, len(self.data_dict)):
                    w_avg[k].add_(self.data_dict[i][k]) 
                w_avg[k] = torch.div(w_avg[k], len(self.data_dict))
            self.mean_tensor_list = w_avg
        lock.release()
        tensor_list={k: v.cpu() for k, v in self.mean_tensor_list.items()}
        time4 = time.time() - time3
        self.ps_time[client_id] += time4
        self.all_time[client_id] += (time.time() - time1)
        if step % endflag == 0:
            a = client_id + 1
            str_log = str(a) + ',' + str(self.all_time[client_id]) + ',' + str(
                self.waiting_time[client_id]) + ',' + str(self.ps_time[client_id])
            self.wait_epoch.append(str_log)
            self.all_time[client_id] = 0
            self.waiting_time[client_id] = 0
            self.ps_time[client_id] = 0
        time5 = time.time() - time1
        return time5, tensor_list

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes parameters remotely.
    def get_param_rrefs(self, cid):
        self.waiting_time[cid - 1] = 0
        self.all_time[cid - 1] = 0
        self.ps_time[cid - 1] = 0
        param_rrefs = {k: v.cpu() for k, v in self.model.state_dict().items()}
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
    train_set = torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True,num_workers=data_worker)  # 分割的数据加载成纯set
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
monitor = []
test1 = []


class Trainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VGG('VGG16').cuda()
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        warm_up_with_cosine_lr = lambda epoch: epoch / warmup_epoch if epoch <= warmup_epoch else 0.5 * (
                math.cos((epoch - warmup_epoch) / (max_epoch - warmup_epoch) * math.pi) + 1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warm_up_with_cosine_lr)
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server)
        self.ps_time = 0
        self.continue_flag =True

    def get_global_param_rrefs(self, cid):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref,
            cid)
        para = {k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(para)

    def wait_finish(self):
        remote_params = remote_method(
            ParameterServer.finish,
            self.param_server_rref,
        )

    def sync(self):
        # tensor_list= [param.data.cpu() for param in self.model.parameters()]
        if train_type == 'BSP':
            grads = []
            for p in self.model.parameters():
                grad = None if p.grad is None else p.grad.data.cpu()
                grads.append(grad)
            self.ps_time, self.continue_flag, remote_params = remote_method(
                ParameterServer.BSP,
                self.param_server_rref,
                grads,
                dic
            )
        elif train_type == 'SSP':
            grads = []
            for p in self.model.parameters():
                grad = None if p.grad is None else p.grad.data.cpu()
                grads.append(grad)
            self.ps_time, self.continue_flag, remote_params = remote_method(
                ParameterServer.SSP,
                self.param_server_rref,
                grads,
                dic
            )
        elif train_type == 'ASP':
            grads = []
            for p in self.model.parameters():
                grad = None if p.grad is None else p.grad.data.cpu()
                grads.append(grad)
            self.ps_time, self.continue_flag, remote_params = remote_method(
                ParameterServer.ASP,
                self.param_server_rref,
                grads,
                dic
            )
        else:
            arg = []
            arg.append(dic['client'])
            arg.append(dic['step'])
            arg.append(dic['end_flag'])
            tensor_list = {k: v.cpu() for k, v in self.model.state_dict().items() if 'weight' in k or 'bias' in k}
            self.ps_time, self.continue_flag, remote_params = remote_method(
                ParameterServer.Local,
                self.param_server_rref,
                arg,
                tensor_list
            )
        para = {k: v.cuda() for k, v in remote_params.items()}
        self.model.load_state_dict(para, False)


criterion = nn.CrossEntropyLoss()


def run_training_loop(rank, n_epochs, train_loader, test_loader):
    # Runs the typical neural network forward + backward + optimizer step, but
    # in a distributed fashion.
    # Build DistributedOptimizer.
    global epoch_time
    global iter_time
    global comm_time
    global train_time
    net = Trainer()
    dic['client'] = rank
    net.get_global_param_rrefs(rank)
    # str_log = 'epoch' + ',' + 'nowTime' + ',' +'train_time:'+ ','+'total compute time:'  +','+ 'total comm time'+','+\
    #            'train_loss:' +',' + 'Accuracy:'
    # write_log(str_log, rank)
    end_flag = len(train_loader)
    dic['end_flag'] = end_flag
    i = 0
    if train_type in type_list:
        for epoch in range(1, n_epochs + 1):
            if not net.continue_flag:
                break
            net.model.train()
            iter_time = 0
            comm_time = 0
            train_time = 0
            epoch_time = 0
            train_loss = 0
            correct = 0
            total = 0
            time1 = time.time()
            for data, target in train_loader:
                if not net.continue_flag:
                    break
                start_time = time.time()
                time2 = time.time()
                data, target = data.cuda(), target.cuda()
                i = i + 1
                dic['step'] = i
                net.optimizer.zero_grad()
                model_output = net.model(data)
                loss = criterion(model_output, target)
                loss.backward()
                #限制梯度大小，避免梯度爆炸，过拟合  [-clip_value,clip_value]
                # torch.nn.utils.clip_grad_norm_(net.model.parameters(), grad_value)
                train_loss += loss.item()
                _, predicted = model_output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                time.sleep((time.time() - time2)*2)
                iter_time += (time.time() - time2)
                time3 = time.time()
                net.sync()
                time.sleep((time.time() - time3 - net.ps_time)*5)
                comm_time += (time.time() - time3 - net.ps_time)
                end_time = time.time()
                epoch_time += (end_time - start_time)
            train_time += (time.time() - time1)
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
            str_log = str(epoch) + ',' + str(nowTime) + ',' + str(train_time) + ',' + str(epoch_time) + ',' + str(iter_time) + \
                      ',' + str(comm_time) + ',' + str(train_loss / len(train_loader)) + ',' + str(100. * correct / total)
            test1.append(str_log)
            get_accuracy(test_loader, net.model, rank, epoch)
            net.scheduler.step()
    else:
        for epoch in range(1, n_epochs + 1):
            if not net.continue_flag:
                break
            net.model.train()
            iter_time = 0
            comm_time = 0
            train_time = 0
            epoch_time = 0
            train_loss = 0
            correct = 0
            total = 0
            time1 = time.time()
            for data, target in train_loader:
                if not net.continue_flag:
                    break
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
                time.sleep((time.time() - time2)*2)
                iter_time += (time.time() - time2)
                if i % H == 0:
                    time3 = time.time()
                    net.sync()
                    time.sleep((time.time() - time3 - net.ps_time)*5)
                    comm_time += (time.time() - time3 - net.ps_time)
                epoch_time += (time.time() - time2)
            net.scheduler.step()
            # time3 = time.time()
            # net.sync()
            # comm_time += (time.time() - time3)
            train_time += (time.time() - time1)
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
            str_log = str(epoch) + ',' + str(nowTime) + ',' + str(train_time) + ',' + str(epoch_time) + ',' + str(iter_time) + \
                      ',' + str(comm_time) + ',' + str(train_loss / len(train_loader)) + ',' + str(100. * correct / total)
            test1.append(str_log)
            # print(f"Epoch {epoch} training complete!")
            # print("Getting accuracy....")
            get_accuracy(test_loader, net.model, rank, epoch)
    f = './log/' + str(train_type) + "_" + "test" + "_" + str(rank) + ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write('epoch' + ',' + 'nowTime' + ',' + 'test_time' + ',' + 'test_loss' + ',' + 'acc' + "\n")
        for key in monitor:
            file.write(str(key) + "\n")
        file.close()
    f = './log/' + str(train_type) + "_" + "train" + "_" + str(rank) + ".csv"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
        file.write(
            'epoch' + ',' + 'nowTime' + ',' + 'train_time'+',' + 'epoch_time' + ',' + 'iter_time' + ',' + 'comm_time' + ',' + 'train_loss' + ',' + 'acc' + "\n")
        for key in test1:
            file.write(str(key) + "\n")
        file.close()
    # get_accuracy(test_loader, net.model, rank, n_epochs)
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
    monitor.append(str_log)


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
        train_loader = partition_dataset()
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = datasets.CIFAR10(
            root='../CIFAR10', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False,num_workers=data_worker)

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
