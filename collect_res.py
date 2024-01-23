import os
import json
import csv
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Collecting training results script', add_help=False)
    parser.add_argument('--logs-path', default='saves/vit-b', type=str)
    return parser

parser = argparse.ArgumentParser('Collecting training results script', parents=[get_args_parser()])
args = parser.parse_args()

# logs_path = 'saves/vit-b'
# logs_path = 'saves/mocov3_vit-b'
logs_path = args.logs_path
keyword = 'consolidator'
archs = ['in21k_deit_base', 'mocov3_deit_base']
datasets = [
    'cifar100', 'caltech101', 'dtd', 'oxford_flowers102', 'svhn', 'sun397', 'oxford_pet', 'patch_camelyon', 'eurosat', 'resisc45', 'diabetic_retinopathy', 'clevr_count', 'clevr_dist', 'dmlab', 'kitti', 'dsprites_loc', 'dsprites_ori', 'smallnorb_azi', 'smallnorb_ele'
]
csv_dict = {}
for arch in archs:
    csv_dict[arch] = {}
    for dataset in datasets:
        csv_dict[arch][dataset] = {
            'consolidator': {
                '384': {
                    'droppath': '-1',
                    'best_acc1': '-1',
                    'best_epoch': '-1',
                    'best_lr': '-1',
                    'best_wd': '-1',
                    'separate_data': {}
                }
            }
        }

data_g1 = ["cifar100", "caltech101", "dtd", "oxford_flowers102", "svhn", "sun397", "oxford_pet"]
data_g2 = ["patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy"]
data_g3 = ["clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"]

res = {}
for p, d, f in os.walk(logs_path):
    for file in f:
        if file != 'log.txt':
            continue
        log_file = os.path.join(p, file)
        dataset = p[len(logs_path+'/'):p.find('_lr')]
        dr = p[p.rfind('_d')+2:]
        if dataset not in res:
            res[dataset] = {'acc': -1.0}
        with open(log_file, 'r') as log:
            text = log.readlines()
            #print('Processing {}'.format(log_file))
            best_acc1 = -1
            best_epoch = -1
            for line in text:
                line_dict = json.loads(line)
                if float(line_dict['test_acc1']) > float(best_acc1):
                    best_acc1 = line_dict['test_acc1']
                    best_epoch = line_dict['epoch']

            print(best_acc1, p)
            if float(best_acc1) > res[dataset]['acc']:
                res[dataset]['acc'] = float(best_acc1)
                #res[dataset]['dr'] = float(dr)

print(res)

avg_g1 = 0.0
for d in data_g1:
    avg_g1 += res[d]['acc']
avg_g1 /= len(data_g1)

avg_g2 = 0.0
for d in data_g2:
    avg_g2 += res[d]['acc']
avg_g2 /= len(data_g2)

avg_g3 = 0.0
for d in data_g3:
    avg_g3 += res[d]['acc']
avg_g3 /= len(data_g3)

print('AVERAGE=', (avg_g1+avg_g2+avg_g3)/3)