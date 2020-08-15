# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse
import numpy as np
import os
import json

import torch

from models import *
from run_manager import RunManager

from config import activation_values
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3')
parser.add_argument('--train', action='store_true')

parser.add_argument('--manual_seed', default=0, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--latency', type=str, default=None)

parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--init_lr', type=float, default=0.05)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='speech_commands', choices=['imagenet', 'speech_commands'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--valid_size', type=int, default=None)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=['None', 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=10)

parser.add_argument('--n_worker', type=int, default=32)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='strong', choices=['normal', 'strong', 'None'])

""" net config """
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument(
    '--net', type=str, default='proxyless_mobile',
    choices=['proxyless_gpu', 'proxyless_cpu', 'proxyless_mobile', 'proxyless_mobile_14']
)
parser.add_argument('--dropout', type=float, default=0)


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach().cpu().numpy()
    return hook


def attach_hooks(net, layer_string=""):
    for name, layer in net._modules.items():
        if name == "act":
            layer.register_forward_hook(get_activation(layer_string))
        elif layer._modules:
            attach_hooks(layer, layer_string + "." + name if layer_string else name)


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # prepare run config
    run_config_path = '%s/run.config' % args.path
    if os.path.isfile(run_config_path):
        # load run config from file
        run_config = json.load(open(run_config_path, 'r'))
        if args.dataset == "speech_commands":
            run_config = SpeechCommandsRunConfig(**run_config)
        elif args.dataset == "imagenet":
            run_config = ImagenetRunConfig(**run_config)
        else:
            raise NotImplementedError
        if args.valid_size:
            run_config.valid_size = args.valid_size
    else:
        raise FileNotFoundError

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    # prepare network
    net_config_path = '%s/net.config' % args.path
    if os.path.isfile(net_config_path):
        # load net from file
        from models import get_net_by_name
        net_config = json.load(open(net_config_path, 'r'))
        net = get_net_by_name(net_config['name']).build_from_config(net_config)
    else:
        raise FileNotFoundError

    # load checkpoints
    best_model_path = '%s/checkpoint/model_best.pth.tar' % args.path
    if os.path.isfile(best_model_path):
        print("Use trained model parameters from model_best.pth.tar")
        init_path = best_model_path
    else:
        print("Fallback to using parameters from init file")
        init_path = '%s/init' % args.path
        if not os.path.isfile(init_path):
            raise FileNotFoundError

    if torch.cuda.is_available():
        checkpoint = torch.load(init_path)
    else:
        checkpoint = torch.load(init_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    net.load_state_dict(checkpoint, strict=True)

    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        net.to(device)
    else:
        raise ValueError

    data_loader = run_config.test_loader
    data_iter = iter(data_loader)

    net.eval()
    attach_hooks(net)

    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        images, labels = data_iter.next()
        images, labels = images.to(device), labels.to(device)
        # compute output
        output = net(images)
        labels = labels.squeeze_()
        loss = criterion(output, labels)
        # measure accuracy and record loss
        acc1 = accuracy(output, labels, topk=(1,))
        acc1 = acc1[0].detach().item()
    print("Test acc = {} %".format(acc1))

    num_plots = len(activation)
    plot_h_w = int(np.ceil(np.sqrt(num_plots)))

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(plot_h_w, plot_h_w)
    fig.tight_layout()
    all_activations = np.empty(1)
    for i, (name, value) in enumerate(activation.items()):
        x = i % plot_h_w
        y = int(np.floor(i / plot_h_w))
        val = value.flatten()
        axs[y][x].hist(val, bins=np.arange(-50, 50, 0.5))
        axs[y][x].axes.yaxis.set_visible(False)
        axs[y][x].axes.set_xlim([-10, 10])
        if "first_conv" in name:
            axs[y][x].set_title("first_conv", fontsize=8)
        elif "blocks" in name:
            name_split = name.split(".")
            axs[y][x].set_title("block " + name_split[1], fontsize=8)
        elif "feature_mix_layer" in name:
            axs[y][x].set_title("feature_mix_layer", fontsize=8)
        else:
            axs[y][x].set_title("unknown", fontsize=8)

        mean = np.mean(val)
        var = np.var(val)
        minim = np.min(val)
        maxim = np.max(val)

        print(name + ":\nmean = {:.2f}, var = {:.2f}, min = {:.2f}, max = {:.2f}".format(mean, var, minim, maxim))

        all_activations = np.concatenate((all_activations, value.flatten()))

    for i in range(len(activation), plot_h_w ** 2):
        x = i % plot_h_w
        y = int(np.floor(i / plot_h_w))
        axs[y][x].axis('off')

    mean = np.mean(all_activations)
    var = np.var(all_activations)
    minim = np.min(all_activations)
    maxim = np.max(all_activations)

    print("Combined" + ":\nmean = {:.2f}, var = {:.2f}, min = {:.2f}, max = {:.2f}".format(mean, var, minim, maxim))

    fig = plt.figure()
    plt.hist(all_activations, bins=np.arange(-50, 50, 0.5))
    axes = plt.gca()
    axes.set_xlim([-10, 10])

    activation = {}


    def replace_relu_with_quantizer(model, n_bit=8):
        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU6):
                setattr(model, child_name, ActivationQuantizer(n_bit, 0, 6))
            else:
                replace_relu_with_quantizer(child, n_bit)

    replace_relu_with_quantizer(net, n_bit=3)
    print(net)
    attach_hooks(net)
    with torch.no_grad():
        # compute output
        output = net(images)
        labels = labels.squeeze_()
        loss = criterion(output, labels)
        # measure accuracy and record loss
        acc1 = accuracy(output, labels, topk=(1,))
        acc1 = acc1[0].detach().item()
    print("Test acc = {} %".format(acc1))


    num_plots = len(activation)
    plot_h_w = int(np.ceil(np.sqrt(num_plots)))

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(plot_h_w, plot_h_w)
    fig.tight_layout()
    all_activations = np.empty(1)
    for i, (name, value) in enumerate(activation.items()):
        x = i % plot_h_w
        y = int(np.floor(i / plot_h_w))
        val = value.flatten()
        axs[y][x].hist(val, bins=np.arange(-50, 50, 0.5))
        axs[y][x].axes.yaxis.set_visible(False)
        axs[y][x].axes.set_xlim([-10, 10])
        if "first_conv" in name:
            axs[y][x].set_title("first_conv", fontsize=8)
        elif "blocks" in name:
            name_split = name.split(".")
            axs[y][x].set_title("block " + name_split[1], fontsize=8)
        elif "feature_mix_layer" in name:
            axs[y][x].set_title("feature_mix_layer", fontsize=8)
        else:
            axs[y][x].set_title("unknown", fontsize=8)

        mean = np.mean(val)
        var = np.var(val)
        minim = np.min(val)
        maxim = np.max(val)

        print(name + ":\nmean = {:.2f}, var = {:.2f}, min = {:.2f}, max = {:.2f}".format(mean, var, minim, maxim))

        all_activations = np.concatenate((all_activations, value.flatten()))

    for i in range(len(activation), plot_h_w ** 2):
        x = i % plot_h_w
        y = int(np.floor(i / plot_h_w))
        axs[y][x].axis('off')

    mean = np.mean(all_activations)
    var = np.var(all_activations)
    minim = np.min(all_activations)
    maxim = np.max(all_activations)

    print("Combined" + ":\nmean = {:.2f}, var = {:.2f}, min = {:.2f}, max = {:.2f}".format(mean, var, minim, maxim))

    fig = plt.figure()
    plt.hist(all_activations, bins=np.arange(-50, 50, 0.1))
    axes = plt.gca()
    axes.set_xlim([-10, 10])

    plt.show()