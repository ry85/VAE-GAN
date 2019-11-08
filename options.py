import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
parser.add_argument('--input_size', default=[3, 64, 64])
parser.add_argument('--beta1', default=0.5, help='Beta1 hyperparam for Adam optimizers')

parser.add_argument('--train_img_dir', type=str, default='../dataset/celeba/train')
parser.add_argument('--train_attr_path', type=str, default='../dataset/celeba/list_attr_celeba_train.txt')
parser.add_argument('--test_img_dir', type=str, default='../dataset/celeba/test')
parser.add_argument('--test_attr_path', type=str, default='../dataset/celeba/list_attr_celeba_test.txt')
parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Pale_Skin', 'Receding_Hairline', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat'])
parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument("--n_epochs",default=15,action="store",type=int,dest="n_epochs")
parser.add_argument("--z_size",default=128,action="store",type=int,dest="z_size")
parser.add_argument("--recon_level",default=3,action="store",type=int,dest="recon_level")
parser.add_argument("--lambda_mse",default=1e-6,action="store",type=float,dest="lambda_mse")
parser.add_argument("--lr",default=3e-4,action="store",type=float,dest="lr")
parser.add_argument("--decay_lr",default=0.75,action="store",type=float,dest="decay_lr")
parser.add_argument("--decay_mse",default=1,action="store",type=float,dest="decay_mse")
parser.add_argument("--decay_margin",default=1,action="store",type=float,dest="decay_margin")
parser.add_argument("--decay_equilibrium",default=1,action="store",type=float,dest="decay_equilibrium")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)