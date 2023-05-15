import cv2
import torch
import os
import mocap.articulate as art
import tqdm
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
sys.path.insert(0, os.path.abspath('mocap'))


totalcapture_dir = r'/media/yxy/OS/yxy/datasets/TotalCapture/'
output_dir = 'results/motions/'


def run(seq_idx, n=0, visualize=False):
    from egolocate import EgoLocate
    net = EgoLocate(slam_setting_file_id=0, visualize=visualize)

    # read video and imu data
    cap = cv2.VideoCapture(os.path.join(totalcapture_dir, 'first_person_head/%d.mp4' % seq_idx))
    accs = torch.load(os.path.join(totalcapture_dir, 'dataset/iacc.pt'))[seq_idx]
    rots = torch.load(os.path.join(totalcapture_dir, 'dataset/irot.pt'))[seq_idx]

    if visualize:
        # read gt result
        gt_pose = torch.load(os.path.join(totalcapture_dir, 'result/gt/pose.pt'))[seq_idx]
        gt_tran = torch.load(os.path.join(totalcapture_dir, 'result/gt/tran.pt'))[seq_idx]
        gt_pose = art.math.axis_angle_to_rotation_matrix(gt_pose).view(-1, 24, 3, 3)

        # read pip result
        pip_pose, pip_tran = torch.load(os.path.join(totalcapture_dir, 'result/pip/%d.pt' % seq_idx))

    # save the result of totalcapture seq argv[1] for the argv[2]th test
    pose, tran = [], []
    tframe = 0
    for i in tqdm.trange(0, accs.shape[0]):
        _, im = cap.read()
        if i % 2 == 0:
            p, t = net.forward_frame(accs[i], rots[i], im, tframe)  # 30fps image
        else:
            p, t = net.forward_frame(accs[i], rots[i])
        pose.append(p)
        tran.append(t)
        tframe += 1 / 60

        if visualize:
            net.set_gt_pose(gt_pose[i], gt_tran[i])
            net.set_pip_pose(pip_pose[i], pip_tran[i])
            net.set_slam_pose(p, t)

    # save results
    pose = torch.stack(pose)
    tran = torch.stack(tran)
    torch.save({'pose': pose, 'tran': tran}, os.path.join(output_dir, 'seq_%d_%d.pt' % (seq_idx, n)))


def evaluate(seq_idx, align_frame=0, use_scale=True, plot=False):
    if seq_idx < 12:
        sid = 'S1'
    elif seq_idx < 24:
        sid = 'S2'
    elif seq_idx < 36:
        sid = 'S3'
    elif seq_idx < 41:
        sid = 'S4'
    else:
        sid = 'S5'

    shape = torch.load(os.path.join(totalcapture_dir, 'dataset/lower_body_length.pt'))[sid]
    scale = (shape['left_hip_length'] + shape['left_upper_leg_length'] + shape['left_lower_leg_length']) / 0.97 if use_scale else 1
    gt_tran = torch.load(os.path.join(totalcapture_dir, 'result/gt/tran.pt'))[seq_idx]
    pip_tran = torch.load(os.path.join(totalcapture_dir, 'result/pip/%d.pt' % seq_idx))[1]
    transpose_tran = torch.load(os.path.join(totalcapture_dir, 'result/transpose/%d.pt' % seq_idx))[1]
    pred_tran_files = glob.glob(os.path.join(output_dir, 'seq_%d_*.pt' % seq_idx))
    pred_tran_files.sort()
    if len(pred_tran_files) == 0:
        return

    # draw all slam result
    err = []
    if plot: plt.title('Seq %d' % seq_idx)
    if plot: plt.ylim(0, 2)
    for i, file in enumerate(pred_tran_files):
        data = torch.load(file)
        data['tran'] = data['tran'] - data['tran'][align_frame]
        dist = (gt_tran - data['tran'] * scale).norm(dim=1)
        if plot: plt.plot(list(range(dist.shape[0])), dist.numpy(), label="%2d  error=%.2fm" % (i, dist.mean()), color='gainsboro')
        err.append(dist.mean().item())

    # highlight slam median
    data = torch.load(pred_tran_files[np.argsort(err)[len(err) // 2]])
    data['tran'] = data['tran'] - data['tran'][align_frame]
    dist = (gt_tran - data['tran'] * scale).norm(dim=1)
    if plot: plt.plot(list(range(dist.shape[0])), dist.numpy(), label="median      error=%.2fm" % dist.mean(), color='black')
    err_slam = dist.mean()

    # draw pip result
    pip_tran = pip_tran - pip_tran[align_frame]
    dist = (gt_tran - pip_tran * scale).norm(dim=1)
    if plot: plt.plot(list(range(dist.shape[0])), dist.numpy(), label="pip             error=%.2fm" % dist.mean())
    err_pip = dist.mean()

    # draw transpose result
    transpose_tran = transpose_tran - transpose_tran[align_frame]
    dist = (gt_tran - transpose_tran * scale).norm(dim=1)
    if plot: plt.plot(list(range(dist.shape[0])), dist.numpy(), label="transpose  error=%.2fm" % dist.mean())
    err_tp = dist.mean()

    print('Seq: %2d' % seq_idx,
          ' \tTP error: %.2f' % err_tp,
          ' \tPIP error: %.2f' % err_pip,
          ' \tEgoLocate error: %.2f +/- %.2f' % (err_slam, np.std(err)))
    if plot: plt.legend()
    if plot: plt.show()

    return err_slam, err_pip, err_tp, np.std(err), len(gt_tran)


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description='evaluate translation on TotalCapture dataset')
    parser.add_argument('--run', nargs=2, type=int, metavar=('SEQ_IDX', 'N'), help='run and save the results of the SEQ_IDX (0~44) sequence for the Nth time')
    parser.add_argument('--visualize', action='store_true', help='visualize the motion comparison during running')
    parser.add_argument('--evaluate', type=str, metavar='MOTION_TYPE', choices=['acting', 'freestyle', 'rom', 'walking', 'all'], help='evaluate translation error for MOTION_TYPE motions')
    parser.add_argument('--plot', action='store_true', help='plot translation error curves for each sequence')
    args = parser.parse_args()

    if args.run is not None:
        run(args.run[0], args.run[1], args.visualize)

    if args.evaluate is not None:
        print('Evaluating ' + args.evaluate + ':')
        print('----------------------------------------------------------------------------------------------')
        seqs = {
            'acting': [0, 1, 2, 12, 13, 14, 24, 25, 26, 36],
            'freestyle': [3, 4, 5, 15, 16, 17, 27, 28, 29, 37, 38, 41, 42],
            'rom': [6, 7, 8, 18, 19, 20, 30, 31, 32, 39, 43],
            'walking': [9, 10, 11, 21, 22, 23, 33, 34, 35, 40, 44],
            'all': list(range(45))
        }[args.evaluate]

        es, ep, et, s, n = 0, 0, 0, 0, 0
        for seq_idx in seqs:   # acting freestyle rom walking full
            err_slam, err_pip, err_tp, std, nframes = evaluate(seq_idx, plot=args.plot)
            es += err_slam * nframes
            ep += err_pip * nframes
            et += err_tp * nframes
            s += std * nframes
            n += nframes
        print('----------------------------------------------------------------------------------------------')
        print('Average:',
              ' \tTP error: %.2f' % (et / n),
              ' \tPIP error: %.2f' % (ep / n),
              ' \tEgoLocate error: %.2f +/- %.2f' % (es / n, s / n))

