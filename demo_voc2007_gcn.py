import argparse
from engine import *
from models import *
from voc import *
import random
import threading
import psutil, GPUtil
import time
import datetime
import pandas as pd

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

ini_rc, ini_wc, ini_rb, ini_wb = psutil.disk_io_counters()[:4]
ini_bs, ini_br = psutil.net_io_counters()[:2]

def sample_metrics(unit="MB"):
    global ini_rc, ini_wc, ini_rb, ini_wb, ini_bs, ini_br

    weight = 1
    if unit == "MB":
        weight = 1024 * 1024
    elif unit == "GB":
        weight = 1024 * 1024 * 1024
    network_stat = psutil.net_io_counters()
    disk_io_stat = psutil.disk_io_counters()
    result = {
        "time": str(datetime.datetime.utcnow()),
        "cpu": psutil.cpu_percent(interval=1),
        "mem": psutil.virtual_memory().used / weight,
        "disk": psutil.disk_usage("/").used / weight,
        "disk_io": {
            "rc": disk_io_stat[0] - ini_rc,
            "wc": disk_io_stat[1] - ini_wc,
            "rb": disk_io_stat[2] - ini_rb,
            "wb": disk_io_stat[3] - ini_wb
        },
        "network": {
            "sent": network_stat.bytes_sent / weight - ini_bs,
            "recv": network_stat.bytes_recv / weight - ini_br
        }
    }
    # if self._use_gpu:
    gpus = GPUtil.getGPUs()
    if len(gpus) > 0:
        result["gpu load"] = gpus[0].load * 100
        result["gpu memutil"] = gpus[0].memoryUtil * 100
    return result

def compute_metrics():
    global running
    running = True
    currentProcess = psutil.Process()

    lst = []
    # start loop
    while running:
        # *measure/store all needed metrics*
        lst.append(sample_metrics())
        time.sleep(1)
    df = pd.DataFrame(lst)
    df.to_csv('demo_voc2007_gcn_metrics.csv', index=False)

def start():
    global t
    # create thread and start it
    t = threading.Thread(target=compute_metrics)
    t.start()

def stop():
    global running
    global t
    # use `running` to stop loop in thread so thread will end
    running = False
    # wait for thread's end
    t.join()

def main_voc2007():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()

    # define dataset
    train_dataset = Voc2007Classification(args.data, 'trainval', inp_name='data/voc/voc_glove_word2vec.pkl')
    val_dataset = Voc2007Classification(args.data, 'test', inp_name='data/voc/voc_glove_word2vec.pkl')

    num_classes = 20

    # load model
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='data/voc/voc_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()
    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/voc2007/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)



if __name__ == '__main__':
    start()
    try:
        main_voc2007()
    finally:
        stop()
