import os
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // args.step_after))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr #/ args.batch_size


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        pred = output.data.max(1)[1]
        acc = pred.eq(target.data).sum().item() * 100.0 / batch_size
        return acc


def setup_data(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    if args.dali:
        try:
            from nvidia.dali.plugin.pytorch import DALIClassificationIterator
            from nvidia.dali.pipeline import Pipeline
            import nvidia.dali.ops as ops
            import nvidia.dali.types as types
        except ImportError:
            raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

        class HybridTrainPipe(Pipeline):
            def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
                super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
                self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
                #let user decide which pipeline works him bets for RN version he runs
                dali_device = 'cpu' if dali_cpu else 'gpu'
                decoder_device = 'cpu' if dali_cpu else 'mixed'
                # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
                # without additional reallocations
                device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
                host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
                self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB, device_memory_padding=device_memory_padding,
                                    host_memory_padding=host_memory_padding, random_aspect_ratio=[0.8, 1.25], random_area=[0.1, 1.0], num_attempts=100)
                self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
                #self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop, crop),
                #image_type=types.RGB, mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
                self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop, crop),
                                                    image_type=types.RGB, mean=[0, 0, 0], std=[255, 255, 255])
                self.coin = ops.CoinFlip(probability=0.5)
                print('DALI "{}" variant, shard id {:d} ({:d} shards)'.format(dali_device, args.local_rank, args.world_size))

            def define_graph(self):
                rng = self.coin()
                self.jpegs, self.labels = self.input(name="Reader")
                images = self.decode(self.jpegs)
                images = self.res(images)
                output = self.cmnp(images.gpu(), mirror=rng)
                return [output, self.labels]

        class HybridValPipe(Pipeline):
            def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
                super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
                self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False)
                self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
                self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
                #self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop, crop),
                #image_type=types.RGB, mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
                self.cmnp = ops.CropMirrorNormalize(device="gpu", output_dtype=types.FLOAT, output_layout=types.NCHW, crop=(crop, crop),
                                                    image_type=types.RGB, mean=[0, 0, 0], std=[255, 255, 255])

            def define_graph(self):
                self.jpegs, self.labels = self.input(name="Reader")
                images = self.decode(self.jpegs)
                images = self.res(images)
                output = self.cmnp(images)
                return [output, self.labels]

        pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=traindir, crop=224, dali_cpu=args.dali_cpu)
        pipe.build()
        train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

        pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=valdir, crop=224, size=256)
        pipe.build()
        val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    else:
        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader


def load_cifar(args):
    print('\n\n\n\t***************** dataset:', args.dataset, '*******************\n\n\n')

    if args.generate_input:  #load entire cifar into RAM?
        print('\n\nloading cifar samples from disk one by one')
    else:
        print('\n\nLoading entire cifar dataset into RAM')

    if args.fp16:
        dtype = np.float16
    else:
        dtype = np.float32

    f = np.load(args.dataset)
    train_inputs = f['arr_0'].reshape(50000, 3, 32, 32).astype(dtype)
    train_labels = f['arr_1']
    test_inputs = f['arr_2'].reshape(10000, 3, 32, 32).astype(dtype)
    test_labels = f['arr_3']
    f.close()

    train_inputs = torch.from_numpy(train_inputs).cuda()
    train_labels = torch.from_numpy(train_labels).cuda()
    test_inputs = torch.from_numpy(test_inputs).cuda()
    test_labels = torch.from_numpy(test_labels).cuda()

    if args.normalize:  #whiten
        mean = np.asarray((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1).astype(dtype)
        std = np.asarray((0.2023, 0.1994, 0.2010)).reshape(1, 3, 1, 1).astype(dtype)

        mean = torch.from_numpy(mean).cuda()
        std = torch.from_numpy(std).cuda()

        test_inputs = test_inputs.sub_(mean).div_(std)
        train_inputs = train_inputs.sub_(mean).div_(std)

    if args.augment:
        # pad train dataset to prepare for random cropping later:
        train_inputs = nn.ZeroPad2d(4)(train_inputs)
        print('Applying random Crop and Flip augmentations\n\n')
    else:
        print('Not augmenting dataset\n\n')

    if args.fp16:
        train_inputs = train_inputs.half()
        test_inputs = test_inputs.half()

    return train_inputs, train_labels, test_inputs, test_labels


def saveargs(args):
    path = args.checkpoint_dir
    if os.path.isdir(path) == False:
        os.makedirs(path)
    with open(os.path.join(path,'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(arg+' '+str(getattr(args,arg))+'\n')

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #nn.init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

'''
#OLD init functions (from pnn):
def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
'''


def init_model(model, args, s=0):

    model.apply(weights_init)
    #model.apply(init_params)
    #print('\n\nUsing old Init method (fan out)\n\n')
    #return

    for n, p in model.named_parameters():
        if 'weight' in n and 'conv' in n:
            if args.weight_init == 'kn':
                if s == 0:
                    print('\n\nInitializing {} to {}, scale param {}\n\n'.format(n, args.weight_init, args.weight_init_scale_conv))
                torch.nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            elif args.weight_init == 'xn':
                if s == 0:
                    print('\n\nInitializing {} to {}, scale param {}\n\n'.format(n, args.weight_init, args.weight_init_scale_conv))
                torch.nn.init.xavier_normal_(p, gain=nn.init.calculate_gain('relu'))
            elif args.weight_init == 'ku':
                if s == 0:
                    print('\n\nInitializing {} to {}, scale param {}\n\n'.format(n, args.weight_init, args.weight_init_scale_conv))
                torch.nn.init.kaiming_uniform_(p, mode='fan_out', nonlinearity='relu')
            elif args.weight_init == 'xu':
                if s == 0:
                    print('\n\nInitializing {} to {}, scale param {}\n\n'.format(n, args.weight_init, args.weight_init_scale_conv))
                torch.nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
            elif args.weight_init == 'ortho':
                if s == 0:
                    print('\n\nInitializing {} to {}, scale param {}\n\n'.format(n, args.weight_init, args.weight_init_scale_conv))
                torch.nn.init.orthogonal_(p, gain=args.weight_init_scale_conv)
            else:
                if s == 0:
                    print('\n\nUNKNOWN init method: NOT Initializing {} to {}, scale param {}\n\n'.format(n, args.weight_init, args.weight_init_scale_conv))

            if args.weight_init_scale_conv != 1.0 and args.weight_init != 'ortho':
                if s == 0:
                    print('\n\nScaling {} weights init by {}\n\n'.format(n, args.weight_init_scale_conv))
                p.data = p.data * args.weight_init_scale_conv

        elif 'linear' in n and 'weight' in n:
            print('\n\nInitializing {} to kaiming normal, scale param {}\n\n'.format(n, args.weight_init_scale_fc))
            nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu')
            if s == 0:
                print('\n\nScaling {} weights init by {}\n\n'.format(n, args.weight_init_scale_fc))
            p.data = p.data * args.weight_init_scale_fc

    if False and args.train_act_max:
        nn.init.constant_(model.act_max1, args.act_max1)
        nn.init.constant_(model.act_max2, args.act_max2)
        nn.init.constant_(model.act_max3, args.act_max3)

    if False and args.train_w_max:
        nn.init.constant_(model.w_min1, -args.w_max1)
        nn.init.constant_(model.w_max1, args.w_max1)


def print_model(model, args, full=False):
    print('\n\n****** Model Configuration ******\n\n')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    if full:
        print('\n\n****** Model Graph ******\n\n')
        for arg in vars(model):
            print(arg, getattr(model, arg))

    print('\n\nModel parameters:\n')
    model_total = 0
    for name, param in model.named_parameters():
        size = param.numel() / 1000.
        print('{}  {}  {:.2f}k'.format(name, list(param.size()), size))
        model_total += size
    print('\n\nModel size: {:.2f}k parameters\n\n'.format(model_total))


def act_fn(act):
    if act == 'relu':
        act_ = nn.ReLU(inplace=False)
    elif act == 'lrelu':
        act_ = nn.LeakyReLU(inplace=True)
    elif act == 'prelu':
        act_ = nn.PReLU()
    elif act == 'rrelu':
        act_ = nn.RReLU(inplace=True)
    elif act == 'elu':
        act_ = nn.ELU(inplace=True)
    elif act == 'selu':
        act_ = nn.SELU(inplace=True)
    elif act == 'tanh':
        act_ = nn.Tanh()
    elif act == 'sigmoid':
        act_ = nn.Sigmoid()
    else:
        print('\n\nActivation function {} is not supported/understood\n\n'.format(act))
        act_ = None
    return act_


def print_values(x, noise, y, unique_masks, n=2):
    np.set_printoptions(precision=5, linewidth=200, threshold=1000000, suppress=True)
    print('\nimage: {}  image0, channel0          {}'.format(list(x.unsqueeze(2).size()), x.unsqueeze(2).data[0, 0, 0, 0, :n].cpu().numpy()))
    print('image: {}  image0, channel1          {}'.format(list(x.unsqueeze(2).size()), x.unsqueeze(2).data[0, 1, 0, 0, :n].cpu().numpy()))
    print('\nimage: {}  image1, channel0          {}'.format(list(x.unsqueeze(2).size()), x.unsqueeze(2).data[1, 0, 0, 0, :n].cpu().numpy()))
    print('image: {}  image1, channel1          {}'.format(list(x.unsqueeze(2).size()), x.unsqueeze(2).data[1, 1, 0, 0, :n].cpu().numpy()))
    if noise is not None:
        print('\nnoise {}  channel0, mask0:           {}'.format(list(noise.size()), noise.data[0, 0, 0, 0, :n].cpu().numpy()))
        print('noise {}  channel0, mask1:           {}'.format(list(noise.size()), noise.data[0, 0, 1, 0, :n].cpu().numpy()))
        if unique_masks:
            print('\nnoise {}  channel1, mask0:           {}'.format(list(noise.size()), noise.data[0, 1, 0, 0, :n].cpu().numpy()))
            print('noise {}  channel1, mask1:           {}'.format(list(noise.size()), noise.data[0, 1, 1, 0, :n].cpu().numpy()))

    print('\nmasks: {} image0, channel0, mask0:  {}'.format(list(y.size()), y.data[0, 0, 0, 0, :n].cpu().numpy()))
    print('masks: {} image0, channel0, mask1:  {}'.format(list(y.size()), y.data[0, 0, 1, 0, :n].cpu().numpy()))
    print('masks: {} image0, channel1, mask0:  {}'.format(list(y.size()), y.data[0, 1, 0, 0, :n].cpu().numpy()))
    print('masks: {} image0, channel1, mask1:  {}'.format(list(y.size()), y.data[0, 1, 1, 0, :n].cpu().numpy()))
    print('\nmasks: {} image1, channel0, mask0:  {}'.format(list(y.size()), y.data[1, 0, 0, 0, :n].cpu().numpy()))
    print('masks: {} image1, channel0, mask1:  {}'.format(list(y.size()), y.data[1, 0, 1, 0, :n].cpu().numpy()))
    print('masks: {} image1, channel1, mask0:  {}'.format(list(y.size()), y.data[1, 1, 0, 0, :n].cpu().numpy()))
    print('masks: {} image1, channel1, mask1:  {}'.format(list(y.size()), y.data[1, 1, 1, 0, :n].cpu().numpy()))
