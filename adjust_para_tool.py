import os
import subprocess
import time

batch_size = [16, 25, 32]
n_hop = [2, 3]
learning_rate = [0.005, 0.003, 0.002, 0.001]
l2_reg = [0.002, 0.001, 0.0005, 0.0002]
loss_weight = [10.0, 8.0, 7.5, 5.0, 2.5, 2.0]
memory_gpu_list = [4]

for a in batch_size:
    for b in n_hop:
        for c in learning_rate:
            for d in l2_reg:
                for e in loss_weight:
                    while True:
                        if a in [25, 32] and c in [0.005, 0.001] and d in [0.001, 0.0005] and e in [10.0, 7.5, 5.0, 2.5, 2.0]:
                            continue
                        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
                        memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]

                        # is there memory gpu idle
                        gpu_id = -1
                        for id, gpu in enumerate(memory_gpu):
                            if id in memory_gpu_list and gpu >= 1500:
                                gpu_id = id
                                break

                        if gpu_id == -1:
                            time.sleep(60 * 5)
                        else:
                            break

                    cmd = ['nohup', 'python', '-u', 'main.py']
                    cmd.extend(['-a', str(a)])
                    cmd.extend(['-b', str(b)])
                    cmd.extend(['-c', str(c)])
                    cmd.extend(['-d', str(d)])
                    cmd.extend(['-e', str(e)])
                    cmd.extend(['-gpu', str(gpu_id)])
                    print('Executing %s' % cmd)

                    with open('result_Lap.txt', 'a') as out:
                        subprocess.Popen(cmd, stdout=out)
                    time.sleep(60 * 3)
