#!/usr/bin/env python
# coding:utf-8

import helper.logger as logger
from models.model import HiAGM
import torch
import sys
from helper.configure import Configure
import os
from data_modules.data_loader import data_loaders
from data_modules.vocab import Vocab
from train_modules.criterions import ClassificationLoss
from train_modules.trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint
import time
from models.proto.cost_matrix import calc_acm
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate,
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")


def train(config, device):
    """
    :param config: helper.configure, Configure Object
    """
    # loading corpus and generate vocabulary
    corpus_vocab = Vocab(config,
                         min_freq=5,
                         max_size=50000)

    # get data
    train_loader, dev_loader, test_loader = data_loaders(config, corpus_vocab)
    
    # build up model
    hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')
    hiagm.to(config.train.device_setting.device)
    # define training objective & optimizer
    criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                                   corpus_vocab.v2i['label'],
                                   recursive_penalty=config.train.loss.recursive_regularization.penalty,
                                   recursive_constraint=config.train.loss.recursive_regularization.flag)
    optimize = set_optimizer(config, hiagm)

    # get epoch trainer
    prototype_dim = config.embedding.token.dimension
    print(corpus_vocab.v2i['label'],"corpus_vocab.v2i['label']")
    num_labels = len(corpus_vocab.v2i['label'])
    # print(corpus_vocab.v2i['label'],"corpus_vocab.v2i['label']")
    # 
    global_prototype_tensor = torch.randn((num_labels+1, prototype_dim), requires_grad=True, device=device).to(device)
    # print(global_prototype_tensor.shape,"global_prototype_tensor")
    ac_matrix = calc_acm(config, corpus_vocab.i2v['label'])
    trainer = Trainer(model=hiagm,
                      criterion=criterion,
                      optimizer=optimize,
                      vocab=corpus_vocab,
                      config=config,
                      global_prototype_tensor=global_prototype_tensor,
                      ac_matrix=ac_matrix)
                      # ,
                      # global_prototype_tensor=global_prototype_tensor)

    # set origin log
    best_epoch = [-1, -1]
    best_performance = [0.0, 0.0]
    model_checkpoint = config.train.checkpoint.dir
    model_name = config.model.type
    wait = 0


    # train
    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        start_time = time.time()
        trainer.train(train_loader,
                      epoch)
        trainer.eval(train_loader, epoch, 'TRAIN')
        print(trainer,"train_loader")
        performance = trainer.eval(dev_loader, epoch, 'DEV')
        # saving best model and check model
        if not (performance['micro_f1'] >= best_performance[0] or performance['macro_f1'] >= best_performance[1]):
            wait += 1
            if wait % config.train.optimizer.lr_patience == 0:
                logger.warning("Performance has not been improved for {} epochs, updating learning rate".format(wait))
                trainer.update_lr()
            if wait == config.train.optimizer.early_stopping:
                logger.warning("Performance has not been improved for {} epochs, stopping train with early stopping"
                               .format(wait))
                break

        if performance['micro_f1'] > best_performance[0]:
            wait = 0
            logger.info('Improve Micro-F1 {}% --> {}%'.format(best_performance[0], performance['micro_f1']))
            best_performance[0] = performance['micro_f1']
            best_epoch[0] = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': hiagm.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, 'best_micro_' + model_name))
        if performance['macro_f1'] > best_performance[1]:
            wait = 0
            logger.info('Improve Macro-F1 {}% --> {}%'.format(best_performance[1], performance['macro_f1']))
            best_performance[1] = performance['macro_f1']
            best_epoch[1] = epoch
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': hiagm.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, 'best_macro_' + model_name))

        if epoch % 10 == 1:
            save_checkpoint({
                'epoch': epoch,
                'model_type': config.model.type,
                'state_dict': hiagm.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimize.state_dict()
            }, os.path.join(model_checkpoint, model_name + '_epoch_' + str(epoch)))
            torch.save(trainer.global_prototype_tensor, os.path.join(model_checkpoint, 'global_prototypes_epoch_{}.pt'.format(epoch)))

        logger.info('Epoch {} Time Cost {} secs.'.format(epoch, time.time() - start_time))

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_micro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=hiagm,
                        config=config,
                        optimizer=optimize)
        trainer.eval(test_loader, best_epoch[0], 'TEST')

    best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_' + model_name)
    if os.path.isfile(best_epoch_model_file):
        load_checkpoint(best_epoch_model_file, model=hiagm,
                        config=config,
                        optimizer=optimize)
        trainer.eval(test_loader, best_epoch[1], 'TEST')
    torch.save(trainer.global_prototype_tensor, os.path.join(model_checkpoint, 'global_prototypes_final.pt'))

    return


if __name__ == "__main__":
    configs = Configure(config_json_file=sys.argv[1])

    if configs.train.device_setting.device == 'cuda':
        os.system('CUDA_VISIBLE_DEVICES=' + str(configs.train.device_setting.visible_device_list))
    else:
        os.system("CUDA_VISIBLE_DEVICES=''")
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    logger.Logger(configs)


    device = configs.train.device_setting.device
    train(configs, device)
