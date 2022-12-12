#!/usr/bin/env bash
echo "开始测试......"
pid=$(netstat -anp | grep :29500 | awk '{print $7}' | awk -F"/" '{ print $1 }');
if [  -n  "$pid"  ];  then
  kill  -9  $pid;
fi
docker exec  node0 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker_ps.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=0 --device="0" '& 
docker exec  node1 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=1 --device="1" '&
docker exec  node2 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=2 --device="2" '& 
docker exec  node3 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=3 --device="3" '& 
docker exec  node4 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=4 --device="4" '& 
docker exec  node5 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=5 --device="5" '& 
docker exec  node6 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=6 --device="6" '& 
docker exec  node7 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=7 --device="7" '& 
docker exec  node8 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=8 --device="8" '&
docker exec  node9 /bin/bash -c 'cd /workspace/pytorch/ASP && python worker.py --world_size=10 --train_type="BSP"  --n_epochs=20  --learning_rate=0.4  --batch_size=128  --rank=9 --device="9" '   
echo "结束测试......"
