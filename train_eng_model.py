import os
import string
import random

from io import BytesIO
from urllib.request import urlretrieve
import tarfile

import torch
from torch import nn, optim
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from models.model.early_exit import Early_conformer
from util.beam_infer import BeamInference
from util.noam_opt import NoamOpt
from util.model_utils import count_parameters, initialize_weights, avg_models
from util.conf import get_args
from util.model_utils import *
from util.tokenizer import *
from data import get_data_loader

import typing
if typing.TYPE_CHECKING:
    from digitalhub.entities.project._base.entity import Project


def download_and_extract(tgzurl, path, filename):
    filepath = path + filename
    print('File download:' + tgzurl)
    urlretrieve(tgzurl, filepath)
    print('File downloaded successfully:' + tgzurl)

    file = tarfile.open(filepath)
    file.extractall(path)
    file.close()
    print('File extracted successfully:' + path)


def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def load_model(args):
    model = Early_conformer(src_pad_idx=args.src_pad_idx,
                            n_enc_exits=args.n_enc_exits,
                            d_model=args.d_model,
                            enc_voc_size=args.enc_voc_size,
                            dec_voc_size=args.dec_voc_size,
                            max_len=args.max_len,
                            d_feed_forward=args.d_feed_forward,
                            n_head=args.n_heads,
                            n_enc_layers=args.n_enc_layers_per_exit,
                            features_length=args.n_mels,
                            drop_prob=args.drop_prob,
                            depthwise_kernel_size=args.depthwise_kernel_size,
                            device=args.device).to(args.device)
    
    model.apply(initialize_weights)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    # print("batch_size:", batch_size, " num_heads:", n_heads, " num_encoder_layers:",
    #     n_enc_layers, "vocab_size:", dec_voc_size, "DEVICE:", device)

    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.n_threads)

    return model


def train(args, model, iterator, optimizer, loss_fn, ctc_loss):

    model.train()
    epoch_loss = 0
    len_iterator = len(iterator)
    print(len_iterator)

    for i, c_batch in enumerate(iterator):
        if len(c_batch) != args.n_batch_split:
            continue

        for batch_0, batch_1, batch_2, batch_3 in c_batch:

            src = batch_0.to(args.device)
            # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28]
            trg = batch_1[:, :-1].to(args.device)
            # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
            trg_expect = batch_1[:, 1:].to(args.device)
            ctc_target_len = batch_2
            valid_lengths = batch_3

            if args.decoder_mode == 'aed':
                att_dec, encoder = model(src, valid_lengths, trg)
                loss_ctc = 0
                loss_ce = 0

                ctc_input_len = torch.full(
                    size=(encoder.size(1),), fill_value=encoder.size(2), dtype=torch.long)

                for dec, enc in zip(att_dec, encoder):
                    loss_ctc += ctc_loss(enc.permute(1, 0, 2), batch_1,
                                         ctc_input_len, ctc_target_len).to(args.device)
                    loss_ce += loss_fn(dec.permute(0, 2, 1), trg_expect)

                del encoder

                loss = args.aed_ce_weight * loss_ce + args.aed_ctc_weight * loss_ctc

            elif args.decoder_mode == 'ctc':
                encoder = model(src, valid_lengths)
                loss_ctc = 0

                ctc_input_len = torch.full(
                    size=(encoder.size(1),), fill_value=encoder.size(2), dtype=torch.long)

                for enc in encoder:
                    loss_ctc += ctc_loss(enc.permute(1, 0, 2), batch_1,
                                         ctc_input_len, ctc_target_len).to(args.device)
                del encoder

                loss = loss_ctc

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item()

            if i % 500 == 0:
                inf = BeamInference(args)
                print("EXPECTED:", args.sp.decode(
                    trg_expect[0].tolist()).lower())
                best_combined = inf.ctc_cuda_predict(
                    emission=enc[0].unsqueeze(0), tokens=args.tokens)
                print("CTC_OUT at [", i, "]:", args.sp.decode(
                    best_combined[0][0].tokens).lower())

        if args.decoder_mode == 'aed':
            print('step: ', round((i / len_iterator) * 100, 2), '% , loss_total: ',
                  loss.item(), 'loss_ce: ', loss_ce.item(), 'loss_ctc: ', loss_ctc.item())
        elif args.decoder_mode == 'ctc':
            print('step: ', round((i / len_iterator) * 100, 2), '% , loss_total: ',
                  loss.item(), 'loss_ctc: ', loss_ctc.item())

    loss_total = epoch_loss / len_iterator

    return loss_total


def run(args, model, total_epoch, best_loss, data_loader, optimizer, loss_fn, ctc_loss):
    
    writer = SummaryWriter()

    loss_prev = 9999999
    nepoch = -1

    moddir = '/data/' + args.save_model_dir + '/'
    os.makedirs(moddir, exist_ok=True)

    best_model = moddir+'{}mod{:03d}-transformer'.format('', nepoch)
    best_lr = moddir+'{}lr{:03d}-transformer'.format('', nepoch)

    if os.path.exists(best_model):
        print('loading model checkpoint:', best_model)
        model.load_state_dict(torch.load(best_model, map_location=args.device))

    if os.path.exists(best_lr):
        print('loading learning rate checkpoint:', best_lr)
        optimizer.load_state_dict(torch.load(best_lr))

    for step in range(nepoch + 1, total_epoch):
        loss_total = train(args=args, model=model,
                           iterator=data_loader, optimizer=optimizer,
                           loss_fn=loss_fn, ctc_loss=ctc_loss)
        writer.add_scalar("Total loss", loss_total, step)
        print("LOSS_TOTAL-", step, ":=", loss_total)

        if loss_total < loss_prev:
            loss_prev = loss_total
            best_model = moddir + 'mod{:03d}-transformer'.format(step)

            print("saving:", best_model)
            torch.save(model.state_dict(), best_model)
            lrate = moddir + 'lr{:03d}-transformer'.format(step)
            print("saving:", lrate)
            torch.save(optimizer.state_dict(), lrate)

        else:
            worst_model = moddir + 'mod{:03d}-transformer'.format(step)
            print("WORST: not saving:", worst_model)

    return best_model

# changed and put the actual passed parameters as default in the train function 
# because passing the arguments is not working properly
def dh_train(project, librispeech_train_dataset: str = "train-clean-100", num_epochs: int = 10, model_name: str = "early-exit-eng-model", base_dir: str = "/shared/"):
    download_dir = '/data/download/'

    try:
        os.mkdir(download_dir)
    except OSError as error:
        print(error)

    # Download and unzip training and test dataset
    train_url = "https://www.openslr.org/resources/12/" + librispeech_train_dataset + ".tar.gz"
    download_and_extract(train_url, download_dir, "train.tar.gz")
    test_url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    download_and_extract(test_url, download_dir, "test.tar.gz")

    print("Getting ARGs...")
    # initialize settings
    args = get_args([], base_dir)
    print(args)
    args.batch_size = 15
    args.n_workers = 3
    args.shuffle = False
    args.decoder_mode = 'ctc'
    args.save_model_dir = 'trained_model'
    args.n_epochs = num_epochs

    #create init model
    print("Loading model...")
    model = load_model(args)
    print("Loading model OK")

    data_loader = get_data_loader(args=args, dataset_path=download_dir)

    # If default, set warmup to length of dataloader
    if args.warmup == -1:
        args.warmup = len(data_loader) * args.n_batch_split

    # Define loss functions (Note: In ctc decoder mode, loss_fn is not used)
    loss_fn = nn.CrossEntropyLoss()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = NoamOpt(args.d_model, args.warmup, 
                        AdamW(params=model.parameters(), lr=0, betas=(0.9, 0.98), eps=args.adam_eps, weight_decay=args.weight_decay))
    
    model_path = run(args=args, model=model, total_epoch=args.n_epochs, best_loss=args.inf,
                     data_loader=data_loader, optimizer=optimizer, loss_fn=loss_fn, ctc_loss=ctc_loss)
    
    if model_name is None:
        model_name = "early-exit-eng-model"

    print("Saving model to datalake...")
    project.log_model(
        name=model_name,
        kind="model",
        source=model_path,
        algorithm="early-exit",
        framework="pythorch"
    )
    print(f"Model {model_name} saved!")


