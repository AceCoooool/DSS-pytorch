import argparse
import os
from dataset import get_loader
from solver import Solver


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config.train_path, config.label_path, config.img_size, config.batch_size,
                                  num_thread=config.num_thread)
        if config.val:
            val_loader = get_loader(config.val_path, config.val_label, config.img_size, config.batch_size,
                                    num_thread=config.num_thread)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_fold, run)): run += 1
        os.mkdir("%s/run-%d" % (config.save_fold, run))
        os.mkdir("%s/run-%d/logs" % (config.save_fold, run))
        # os.mkdir("%s/run-%d/images" % (config.save_fold, run))
        os.mkdir("%s/run-%d/models" % (config.save_fold, run))
        config.save_fold = "%s/run-%d" % (config.save_fold, run)
        if config.val:
            train = Solver(train_loader, val_loader, None, config)
        else:
            train = Solver(train_loader, None, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader = get_loader(config.test_path, config.test_label, config.img_size, config.batch_size, mode='test',
                                 num_thread=config.num_thread)
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, None, test_loader, config)
        test.test(100)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    data_root = os.path.join(os.path.expanduser('~'), 'data')
    train_path = os.path.join(data_root, 'ECSSD/images')
    label_path = os.path.join(data_root, 'ECSSD/ground_truth_mask')
    vgg_path = './weights/vgg16_feat.pth'

    val_path = os.path.join(data_root, 'ECSSD/val_images')
    val_label = os.path.join(data_root, 'ECSSD/val_ground_truth_mask')
    test_path = os.path.join(data_root, 'ECSSD/test_images')
    test_label = os.path.join(data_root, 'ECSSD/test_ground_truth_mask')
    # test_path = os.path.join(data_root, 'ECSSD/images')
    # test_label = os.path.join(data_root, 'ECSSD/ground_truth_mask')
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--train_path', type=str, default=train_path)
    parser.add_argument('--label_path', type=str, default=label_path)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val', type=bool, default=False)
    parser.add_argument('--val_path', type=str, default=val_path)
    parser.add_argument('--val_label', type=str, default=val_label)
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--epoch_val', type=int, default=20)
    parser.add_argument('--epoch_save', type=int, default=20)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--test_path', type=str, default=test_path)
    parser.add_argument('--test_label', type=str, default=test_label)
    parser.add_argument('--model', type=str, default='./weights/best.pth')
    parser.add_argument('--test_fold', type=str, default='./results/test')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
