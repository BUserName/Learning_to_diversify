import argparse

import torch
import os
from torch import nn
from torch.nn import functional as F
from data import data_helper
from data.data_helper import available_datasets
from models import model_factory

from utils.Logger import Logger
from utils.util import *

from models.augnet import AugNet
from models.caffenet import caffenet
from models.resnet import resnet18
from utils.contrastive_loss import SupConLoss

from torchvision import transforms

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0., type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--visualization", default=False, type=bool)
    parser.add_argument("--epochs_min", type=int, default=1,
                        help="")
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--ckpt", default="logs/model", type=str)
    #
    parser.add_argument("--alpha1", default=1, type=float)
    parser.add_argument("--alpha2", default=1, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--lr_sc", default=10, type=float)
    parser.add_argument("--task", default='PACS', type=str)

    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.counterk=0

        # Caffe Alexnet for singleDG task, Leave-one-out PACS DG task.
        # self.extractor = caffenet(args.n_classes).to(device)
        self.extractor = resnet18(classes=args.n_classes).to(device)
        self.convertor = AugNet(1).cuda()

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=False)
        if len(self.args.target) > 1:
            self.target_loader = data_helper.get_multiple_val_dataloader(args, patches=False)
        else:
            self.target_loader = data_helper.get_val_dataloader(args, patches=False)

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)

        # Get optimizers and Schedulers, self.discriminator
        self.optimizer = torch.optim.SGD(self.extractor.parameters(), lr=self.args.learning_rate, nesterov=True, momentum=0.9, weight_decay=0.0005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.args.epochs *0.8))

        self.convertor_opt = torch.optim.SGD(self.convertor.parameters(), lr=self.args.lr_sc)

        self.n_classes = args.n_classes
        self.centroids = 0
        self.d_representation = 0
        self.flag = False
        self.con = SupConLoss()
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None


    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        self.extractor.train()
        tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for it, ((data, _, class_l), _, idx) in enumerate(self.source_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            # Stage 1
            self.optimizer.zero_grad()

            # Aug
            inputs_max = tran(torch.sigmoid(self.convertor(data)))
            inputs_max = inputs_max * 0.6 + data * 0.4
            data_aug = torch.cat([inputs_max, data])
            labels = torch.cat([class_l, class_l])

            # forward
            logits, tuple = self.extractor(data_aug)

            # Maximize MI between z and z_hat
            emb_src = F.normalize(tuple['Embedding'][:class_l.size(0)]).unsqueeze(1)
            emb_aug = F.normalize(tuple['Embedding'][class_l.size(0):]).unsqueeze(1)
            con = self.con(torch.cat([emb_src, emb_aug], dim=1), class_l)

            # Likelihood
            mu = tuple['mu'][class_l.size(0):]
            logvar = tuple['logvar'][class_l.size(0):]
            y_samples = tuple['Embedding'][:class_l.size(0)]
            likeli = -loglikeli(mu, logvar, y_samples)

            # Total loss & backward
            class_loss = criterion(logits, labels)
            loss = class_loss + self.args.alpha2*likeli + self.args.alpha1*con
            loss.backward()
            self.optimizer.step()
            _, cls_pred = logits.max(dim=1)

            # STAGE 2
            inputs_max =tran(torch.sigmoid(self.convertor(data, estimation=True)))
            inputs_max = inputs_max * 0.6 + data * 0.4
            data_aug = torch.cat([inputs_max, data])

            # forward with the adapted parameters
            outputs, tuples = self.extractor(x=data_aug)

            # Upper bound MI
            mu = tuples['mu'][class_l.size(0):]
            logvar = tuples['logvar'][class_l.size(0):]
            y_samples = tuples['Embedding'][:class_l.size(0)]
            div = club(mu, logvar, y_samples)
            # div = criterion(outputs, labels)

            # Semantic consistency
            e = tuples['Embedding']
            e1 = e[:class_l.size(0)]
            e2 = e[class_l.size(0):]
            dist = conditional_mmd_rbf(e1, e2, class_l, num_class=self.args.n_classes)

            # Total loss and backward
            self.convertor_opt.zero_grad()
            (dist + self.args.beta * div).backward()
            self.convertor_opt.step()
            self.logger.log(it, len(self.source_loader),
                            {"class": class_loss.item(),
                             "AUG:": torch.sum(cls_pred[:class_l.size(0)] == class_l.data).item() / class_l.shape[0],
                             },
                            {"class": torch.sum(cls_pred[class_l.size(0):] == class_l.data).item()},
                            class_l.shape[0])

        del loss, class_loss, logits

        self.extractor.eval()
        with torch.no_grad():
            if len(self.args.target) > 1:
                avg_acc = 0
                for i, loader in enumerate(self.target_loader):
                    total = len(loader.dataset)

                    class_correct = self.do_test(loader)

                    class_acc = float(class_correct) / total
                    self.logger.log_test('test', {"class": class_acc})

                    avg_acc += class_acc
                avg_acc = avg_acc / len(self.args.target)
                print(avg_acc)
                self.results["test"][self.current_epoch] = avg_acc
            else:
                for phase, loader in self.test_loaders.items():
                    if self.args.task == 'HOME' and phase == 'val':
                        continue
                    total = len(loader.dataset)

                    class_correct = self.do_test(loader)

                    class_acc = float(class_correct) / total
                    self.logger.log_test(phase, {"class": class_acc})
                    self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, nouse, class_l), _, _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)


            z = self.extractor(data, train=False)[0]


            _, cls_pred = z.max(dim=1)

            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        current_high = 0
        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch(self.current_epoch)
            self.scheduler.step()
            if self.results["test"][self.current_epoch] > current_high:
                print('Saving Best model ...')
                torch.save(self.extractor.state_dict(), os.path.join('/home/zijian/Desktop/DG/L2D/large_images/logs/model/', 'best_'+self.args.target[0]))
                current_high = self.results["test"][self.current_epoch]
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = test_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best test epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger

    def do_eval(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        current_high = 0
        self.logger.new_epoch(self.scheduler.get_lr())
        self.extractor.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)

                class_correct = self.do_test(loader)

                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"class": class_acc})
                self.results[phase][0] = class_acc

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = test_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best test epoch: %g" % (
            val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger

def main():
    args = get_args()

    if args.task == 'PACS':
        args.n_classes = 7
        args.source = ['art_painting', 'cartoon', 'sketch']
        args.target = ['photo']
        # args.source = ['art_painting', 'photo', 'cartoon']
        # args.target = ['sketch']
        # args.source = ['art_painting', 'photo', 'sketch']
        # args.target = ['cartoon']
        # args.source = ['photo', 'cartoon', 'sketch']
        # args.target = ['art_painting']
        # --------------------- Single DG
        # args.source = ['photo']
        # args.target = ['art_painting', 'cartoon', 'sketch']

    elif args.task == 'VLCS':
        args.n_classes = 5
        # args.source = ['CALTECH', 'LABELME', 'SUN']
        # args.target = ['PASCAL']
        args.source = ['LABELME', 'SUN', 'PASCAL']
        args.target = ['CALTECH']
        # args.source = ['CALTECH', 'PASCAL', 'LABELME' ]
        # args.target = ['SUN']
        # args.source = ['CALTECH', 'PASCAL', 'SUN']
        # args.target = ['LABELME']

    elif args.task == 'HOME':
        args.n_classes = 65
        # args.source = ['real', 'clip', 'product']
        # args.target = ['art']
        # args.source = ['art', 'real', 'product']
        # args.target = ['clip']
        # args.source = ['art', 'clip', 'real']
        # args.target = ['product']
        args.source = ['art', 'clip', 'product']
        args.target = ['real']
    # --------------------------------------------
    print("Target domain: {}".format(args.target))
    fix_all_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    trainer = Trainer(args, device)
    if not args.eval:
        trainer.do_training()
    else:
        trainer.extractor.load_state_dict(torch.load(''))
        trainer.do_eval()



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
