import subprocess as sp
import os
from threading import Thread, Timer
import sched, time


def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]

    return memory_use_values


def get_gpu_util():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]

    return memory_use_values


def trace_gpu(t, file, end):
    begin = time.time()
    interval=0
    last = begin
    while True:
        current = time.time() - begin
        interval = time.time()-last

        if(interval<t):
            continue
        else:
            file.write("Time:" + str(current) + " memory usage: " + str(get_gpu_memory()) + "\n")
            file.write("Time:" + str(current) + " utilization: " + str(get_gpu_util()) + "\n")
            last = time.time()
        if (current > end):
            break



f = open(r"/Data/GPU-logging/log.txt", "w")

begin_time = time.time()

trace_gpu(t=5.0, file=f, end=10)

