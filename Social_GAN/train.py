import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import os
import sys
from loader import data_loader
from functions import int_tuple, get_total_norm
from functions import relative_to_abs, get_dset_path
from models import gan_g_loss, gan_d_loss, l2_loss, cal_l2_losses, cal_ade, cal_fde
from models import displacement_error, final_displacement_error
from models import TrajectoryGenerator, TrajectoryDiscriminator
from collections import defaultdict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='stanford', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=14, type=int)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--num_epochs', default=300, type=int)
# Model Options
parser.add_argument('--embedding_dim', default=16, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--batch_norm', default=False)
parser.add_argument('--mlp_dim', default=64, type=int)
# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int)
parser.add_argument('--decoder_h_dim_g', default=32, type=int)
parser.add_argument('--noise_dim', default=(14,), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global')
parser.add_argument('--clipping_threshold_g', default=1.0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)
# Pool Net Option
parser.add_argument('--bottleneck_dim', default=32, type=int)
# Discriminator Options
parser.add_argument('--d_type', default='global', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)
# Loss Options
parser.add_argument('--best_k', default=20, type=int)
# Output
parser.add_argument('--checkpoint_every', default=1, type=int)
parser.add_argument('--save_model_every', default=20, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--model_name', default="new", type=str)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'val')

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    generator = TrajectoryGenerator(
        obs_len=args.obs_len, 
        pred_len=args.pred_len, 
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g, 
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim, 
        num_layers=args.num_layers, 
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        bottleneck_dim=args.bottleneck_dim,
        batch_norm=args.batch_norm)

    generator.apply(init_weights)
    generator.cuda().train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.cuda().train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )

    epoch = 0
    if args.model_name == "new":
        print("Training New Model")
        SavedModel = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'norm_g': [],
            'norm_d': [],
            'epoch': [],
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
        }
    else:
        print("Training on Saved Model")
        SavedModel = torch.load("./" + args.model_name + ".pt")
        generator.load_state_dict(SavedModel['g_state'])
        discriminator.load_state_dict(SavedModel['d_state'])
        optimizer_g.load_state_dict(SavedModel['g_optim_state'])
        optimizer_d.load_state_dict(SavedModel['d_optim_state'])
        epoch = (SavedModel['epoch'])[-1]
    start = epoch
    
    while epoch < args.num_epochs + start:
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        SavedModel['epoch'].append(epoch)
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if d_steps_left > 0:
                losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                SavedModel['norm_g'].append(get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g)
                SavedModel['norm_g'].append(get_total_norm(generator.parameters()))
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            d_steps_left = args.d_steps
            g_steps_left = args.g_steps


        for k, v in sorted(losses_d.items()):
            logger.info('  [D] {}: {:.3f}'.format(k, v))
            SavedModel['D_losses'][k].append(v)
        for k, v in sorted(losses_g.items()):
            logger.info('  [G] {}: {:.3f}'.format(k, v))
            SavedModel['G_losses'][k].append(v)
        

        if epoch % args.checkpoint_every == 0:
            logger.info('Checking stats on val ...')
            metrics_val = check_accuracy(
                args, val_loader, generator, discriminator, d_loss_fn
            )
            logger.info('Checking stats on train ...')
            metrics_train = check_accuracy(
                args, train_loader, generator, discriminator,
                d_loss_fn, limit=True
            )

            for k, v in sorted(metrics_val.items()):
                logger.info('  [val] {}: {:.3f}'.format(k, v))
                SavedModel['metrics_val'][k].append(v)
            for k, v in sorted(metrics_train.items()):
                logger.info('  [train] {}: {:.3f}'.format(k, v))
                SavedModel['metrics_train'][k].append(v)
            
        if epoch % args.save_model_every == 0:
            logger.info('Saving Model ...')
            SavedModel['g_state'] = generator.state_dict()
            SavedModel['g_optim_state'] = optimizer_g.state_dict()
            SavedModel['d_state'] = discriminator.state_dict()
            SavedModel['d_optim_state'] = optimizer_d.state_dict()

            checkpoint_path = os.path.join(os.getcwd(), 'SocialGANModel_%d.pt' % epoch)
            torch.save(SavedModel, checkpoint_path)

def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)
        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        g_l2_loss_rel.append(l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)

    g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
    for start, end in seq_start_end.data:
        _g_l2_loss_rel = g_l2_loss_rel[start:end]
        _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
        _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
            loss_mask[start:end])
        g_l2_loss_sum_rel += _g_l2_loss_rel

    losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
    loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy(
    args, loader, generator, discriminator, d_loss_fn, limit=False
):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            
            #print(total_traj)
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
