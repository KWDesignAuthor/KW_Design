import logging
import sys

import json
import torch
import os.path as osp
import argparse
# from config import create_parser
import time
import warnings
warnings.filterwarnings('ignore')
from opencpd.utils.recorder import Recorder
from opencpd.utils.main_utils import print_log, output_namespace, set_seed, check_dir, load_config, get_dataset
import wandb
from datetime import datetime
from opencpd.methods import method_maps
import os
import copy


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=10, type=int, help='Interval in batches between display of training metrics')
    parser.add_argument('--ex_name', default='CATH4.3/ESMIF', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--no_wandb', default=1, type=int)
    
    # CATH
    # dataset parameters
    parser.add_argument('--data_name', default='MPNN', choices=['MPNN', 'PDB', 'CATH4.2', 'TS50', 'CATH4.3'])
    parser.add_argument('--data_root', default='/data/')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--score_thr', default=70.0, type=float)


    # method parameters
    parser.add_argument('--method', default='PiFold_CA', choices=['AlphaDesign', 'PiFold', 'KWDesign', 'GraphTrans', 'StructGNN', 'GVP', 'GCA', 'ProteinMPNN', 'ESMIF', 'PiFold_CA'])
    parser.add_argument('--config_file', '-c', default=None, type=str)
    
    # Training parameters
    
    parser.add_argument('--epoch', default=20, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate')
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--augment_eps', default=0.00, type=float, help='augment_eps')
    parser.add_argument('--removeTS', default=0, type=int, help='remove training and validation samples that have 30+% similarity to TS50 and TS500')
    
    parser.add_argument('--num_encoder_layers', default=10, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    
    return parser.parse_args()


class Exp:
    def __init__(self, args, show_params=True):
        self.args = args
        self.config = args.__dict__
        self.device = self._acquire_device()
        self.total_step = 0
        self._preparation()
        if show_params:
            print_log(output_namespace(self.args))
    
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0')
            print('Use GPU:',device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _preparation(self):
        set_seed(self.args.seed)
        # torch.use_deterministic_algorithms(True)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        
        # build the method
        if self.args.method!="KWDesign":
            self._build_method()

    def _build_method(self):
        steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, steps_per_epoch)

    def _get_data(self): 
        self.train_loader, self.valid_loader, self.test_loader = get_dataset(self.config)

    
    def _save(self, name=''):
        if self.args.method=='KWDesign':
            torch.save({key:val for key,val in self.method.model.state_dict().items() if "GNNTuning" in key}, osp.join(self.checkpoints_path, name + '.pth'))
        else:
            torch.save(self.method.model.state_dict(), osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, epoch):
        self.method.model.load_state_dict(torch.load(osp.join(self.checkpoints_path, str(epoch) + '.pth')), strict=False)
    
    def train_KWDesign(self):
        self.args.patience = 5
        self.args.epoch = 5
        recycle_n = self.args.recycle_n
        for cycle in range(1, recycle_n+1):
            self.args.recycle_n = cycle
            current_pth = osp.join(self.args.res_dir, self.args.ex_name, "checkpoints", f"msa{self.args.msa_n}_recycle{self.args.recycle_n}_epoch{self.args.load_epoch}.pth")
            if os.path.exists(current_pth):
                continue
            else:
                self._build_method()
                self.train()
    
    def train(self):
        recorder = Recorder(self.args.patience, verbose=True)
        for epoch in range(self.args.epoch):
            if self.args.method=='KWDesign':
                prev_memory_len = len(self.method.model.memo_pifold.memory)
            train_loss, train_perplexity = self.method.train_one_epoch(self.train_loader)

            if epoch % self.args.log_step==0:
                with torch.no_grad():
                    valid_loss, valid_perplexity = self.valid()
                    if self.args.method=='KWDesign':
                        self._save(name=f"msa{self.args.msa_n}_recycle{self.args.recycle_n}_epoch{epoch}")
                        if not os.path.exists(self.args.memory_path):
                            torch.save({"memo_pifold":self.method.model.memo_pifold.memory, "memo_esmif":self.method.model.memo_esmif.memory} , self.args.memory_path)
                        
                        new_memory_len = len(self.method.model.memo_pifold.memory)
                        if new_memory_len!=prev_memory_len:
                            torch.save({"memo_pifold":self.method.model.memo_pifold.memory, "memo_esmif":self.method.model.memo_esmif.memory} , self.args.memory_path)
                    else:
                        if epoch > self.args.epoch-10:
                            self._save(name=str(epoch))
                
                print_log('Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Train Perp: {3:.4f} Valid Loss: {4:.4f} Valid Perp: {5:.4f}\n'.format(epoch + 1, len(self.train_loader), train_loss, train_perplexity, valid_loss, valid_perplexity))
                
                if not self.args.no_wandb:
                    wandb.log({"valid_perplexity": valid_perplexity})

                if self.args.method=='KWDesign':
                    recorder(valid_loss, {key:val for key,val in self.method.model.state_dict().items() if "GNNTuning" in key}, self.path)
                else:
                    recorder(valid_loss, self.method.model.state_dict(), self.path)
                    
                if recorder.early_stop:
                    print("Early stopping")
                    logging.info("Early stopping")
                    break
            
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        self.method.model.load_state_dict(torch.load(best_model_path), strict=False)

    def valid(self):
        valid_loss, valid_perplexity = self.method.valid_one_epoch(self.valid_loader)

        print_log('Valid Perp: {0:.4f}'.format(valid_perplexity))
        
        return valid_loss, valid_perplexity

    def test(self):
        test_perplexity, test_recovery = self.method.test_one_epoch(self.test_loader)
        print_log('Test Perp: {0:.4f}, Test Rec: {1:.4f}\n'.format(test_perplexity, test_recovery))
        if not self.args.no_wandb:
            wandb.log({"test_perplexity": test_perplexity,
                       "test_acc": test_recovery})

        return test_perplexity, test_recovery


def main():
    pass

if __name__ == '__main__':
    main()