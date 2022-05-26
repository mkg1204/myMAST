import logger
import argparse
import os, time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn
import numpy as np

from models.mast import MAST
from datas.ytb_dataset import YTB_Test

from functional.utils.io import imwrite_indexed

def main():
    args.training = False
    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath)
    log = logger.setup_logger(args.savepath + '/benchmark.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))
    
    DataLoader = YTB_Test(args.datapath, args.annopath, args.metapath, args.size)
    seq_nums = len(DataLoader)
    seq_each_group = seq_nums // args.groups
    start_seq_id = seq_each_group * args.groupid
    end_seq_id = start_seq_id + seq_each_group if args.groupid < args.groups-1 else seq_nums
    DataLoader.seq_list = DataLoader.seq_list[start_seq_id:end_seq_id]
    DataLoader = torch.utils.data.DataLoader(
        DataLoader,
        batch_size=1, shuffle=False,num_workers=0,drop_last=False
    )
    #seq_nums = len(DataLoader)
    #seq_each_group = seq_nums // args.groups
    #start_seq_id = seq_each_group * args.groupid
    #end_seq_id = start_seq_id + seq_each_group if args.groupid < args.groups-1 else seq_nums
    model = MAST(args)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    log.info("start:{}, end:{}, all:{}".format(start_seq_id, end_seq_id, len(DataLoader)))
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            log.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            log.info("=> No checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('=> No checkpoint file. Start from scratch.')
    model = nn.DataParallel(model).cuda()

    start_full_time = time.time()

    test(DataLoader, model, log)

    log.info('full testing time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def test(dataloader, model, log):
    model.eval()

    torch.backends.cudnn.benchmark = True

    n_b = len(dataloader)

    log.info("Start testing")
    for b_i, (images_rgb, annotations, new_objs, seq_info) in enumerate(dataloader):
        images_rgb = [r.cuda() for r in images_rgb]
        annotations = [q.cuda() for q in annotations]

        N = len(images_rgb)
        outputs = [annotations[0].contiguous()]

        for i in range(N-1):
            mem_gap = 2
            # ref_index = [i]
            if args.ref == 0:
                ref_index = list(filter(lambda x: x <= i, [0, 5])) + list(filter(lambda x:x>0,range(i,i-mem_gap*3,-mem_gap)))[::-1]
                ref_index = sorted(list(set(ref_index)))
            elif args.ref == 1:
                ref_index = [0] + list(filter(lambda x: x > 0, range(i, i - mem_gap * 3, -mem_gap)))[::-1]
            elif args.ref == 2:
                ref_index = [i]
            else:
                raise NotImplementedError

            rgb_0 = [images_rgb[ind] for ind in ref_index]  # 参考帧的图片
            rgb_1 = images_rgb[i+1]                         # 当前帧的图片

            anno_0 = [outputs[ind] for ind in ref_index]    # 参考帧的预测结果
            anno_1 = annotations[i+1]                       # 当前帧的标注

            _, _, h, w = anno_0[0].size()

            # max_class = anno_1.max()

            with torch.no_grad():
                _output = model(rgb_0, anno_0, rgb_1, ref_index, i+1)
                _output = F.interpolate(_output, (h,w), mode='bilinear')

                output = torch.argmax(_output, 1, keepdim=True).float()

                # 将当前帧新出现的目标叠加在上一帧的结果上
                if len(new_objs[i+1]) > 0:
                    for new_obj_id in new_objs[i+1]:
                        new_obj_id_ = (torch.ones_like(output) * new_obj_id.cuda().float()).long()
                        output[anno_1 == new_obj_id_] = new_obj_id

                outputs.append(output)

            ###
            folder = os.path.join(args.savepath,'benchmark')
            if not os.path.exists(folder): os.mkdir(folder)

            output_folder = os.path.join(folder, seq_info["seq_name"][0].strip())

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            pad =  ((0,0), (0,0))
            if i == 0:
                # output first mask
                output_file = os.path.join(output_folder, seq_info['seq_imgs'][0][0].replace('.jpg', '.png'))
                out_img = anno_0[0][0, 0].cpu().numpy().astype(np.uint8)

                out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
                imwrite_indexed(output_file, out_img, resize_size=seq_info["seq_size"])

            output_file = os.path.join(output_folder, seq_info['seq_imgs'][i+1][0].replace('.jpg', '.png'))
            out_img = output[0, 0].cpu().numpy().astype(np.uint8)
            out_img = np.pad(out_img, pad, 'edge').astype(np.uint8)
            imwrite_indexed(output_file, out_img, resize_size=seq_info["seq_size"])


        log.info('[{}/{}] {}'.format( b_i, n_b, seq_info["seq_name"]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAST')

    # Data options
    parser.add_argument('--ref', type=int, default=0)

    parser.add_argument('--datapath', type=str, default='/datasets/Youtube-VOS')
    parser.add_argument('--annopath', type=str, default='Annotations')
    parser.add_argument('--metapath', type=str, default='meta18.json')
    parser.add_argument('--savepath', type=str, default='results/test')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint file to resume')
    parser.add_argument('--size', type=int, default=None, help='Width to resize')
    parser.add_argument('--groups', type=int, default=1)
    parser.add_argument('--groupid', type=int, default=0)

    args = parser.parse_args()

    main()
