import os
import glob
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

RANDOM_SEED = 1
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--test', type=eval, default=True)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--ntotal', type=int, default=500)  # total number of points in spiral
parser.add_argument('--nsample', type=int, default=100)  # number of observed points for training
parser.add_argument('--ntest', type=int, default=100)  # number of testing points
parser.add_argument('--noise', type=float, default=0.1)  # gaussian noise stdv
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=3)
parser.add_argument('--train_dir', type=str, default='./checkpoint')  # pretrained
parser.add_argument('--save', type=str, default='./latent_ode_out')
parser.add_argument('--method', type=str, default='dopri5')  # euler, dopri5_err
parser.add_argument('--l1', type=float, default=0)  # lambda for l1 regularization 0.5
parser.add_argument('--l2', type=float, default=0)  # l2 regularization (Adam.weight_decay) 0.01
parser.add_argument('--dopri_lambda', type=float, default=0)  # dopri error term as regularizer
parser.add_argument('--kinetic_lambda', type=float, default=0)  # kinetic energy term as regularizer
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq_ import odeint_adjoint as odeint
else:
    from torchdiffeq_ import odeint_err as odeint_err
    from torchdiffeq_ import odeint as odeint

from torchdiffeq_ import RegularizedODEfunc, quadratic_cost

# from torchdiffeq_ import odeint, odeint_adjoint

def generate_spiral2d(nspiral=1000,  # 1000 spirals
                      ntotal=500,  # total number of datapoints per spiral
                      nsample=100,  # sampled(observed) at equally-spaced timesteps
                      ntest=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,  # guassian noise for reality
                      a=0.,
                      b=1.,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),  # whole spiral
      and fourth element is timestamps of size (nsample,)  # sampled spiral
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)  # evenly spaced 500 timestamps
    samp_ts = orig_ts[:nsample]  # time points for which to solve
    test_ts = -orig_ts[:ntest][::-1].copy()
    test_ts[-1] = -test_ts[-1]  # revert -0 to 0

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)  # original trajectory of clockwise spiral

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)  # original trajectory of counter-clockwise spiral

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig(args.save + '/ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format(args.save + '/ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []  # whole 500 points of all 1000 spirals
    samp_trajs = []  # sampled 100 points of all 1000 spirals
    test_trajs = []  # testing points until t0
    for _ in range(nspiral):  # generate 1000 spirals
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # uniformly select rotation (clockwise | counter-clockwise)
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)  # ground-truth spiral

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()  # 100 points starting from t0_idx
        samp_traj += npr.randn(*samp_traj.shape) * noise_std  # add guassian noise for observation reality
        samp_trajs.append(samp_traj)

        test_traj = orig_traj[t0_idx - ntest:t0_idx, :][::-1].copy()  # 100 previous points from t0_idx
        # test_traj += npr.randn(*test_traj.shape) * noise_std  # add guassian noise for observation reality
        test_trajs.append(test_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    test_trajs = np.stack(test_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts, test_trajs, test_ts


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)  # one hidden layer network
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):  # 2-dimensional spirals
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):  # 2-dimensional spirals
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)  # one hidden layer network
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':
    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    nspiral = 1000
    start = 0.
    stop = 6 * np.pi  # 19
    noise_std = args.noise # .03
    a = 0.
    b = .3
    # ntotal = 1000

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # generate toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts, test_trajs, test_ts = generate_spiral2d(
        nspiral=nspiral,
        ntotal=args.ntotal,  # total number of datapoints per spiral
        nsample=args.nsample,  # sampled(observed) at equally-spaced timesteps
        ntest=args.ntest,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a, b=b
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)  # (1000, 500, 2) of ground-truth
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)  # (1000, 100, 2) of sampled points from orig_trajs
    samp_ts = torch.from_numpy(samp_ts).float().to(device)  # first 100 timestamps to sample points at

    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    # func = RegularizedODEfunc(odefunc, quadratic_cost)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr) #, weight_decay=args.l2)  # l2 regularization
    loss_meter = RunningAverageMeter()

    if args.train_dir is not None:  # pretrained
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            samp_trajs = checkpoint['samp_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']
            logger.info('Loaded ckpt from {}'.format(ckpt_path))

    try:
        batch_time_meter = RunningAverageMeter()
        logger.info('*' * 15 + 'start training' + '*' * 15)  # interpolation
        epoch_nfe = []
        best_rmse = {}
        for itr in range(1, args.niters + 1):  # 2000
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)  # nhidden=25, nbatch=1
            for t in reversed(range(samp_trajs.size(1))):  # 100 (1000, 100, 2)
                obs = samp_trajs[:, t, :]  # (1000, 2). t'th observed(sampled) point from 1000 spirals
                out, h = rec.forward(obs, h)  # recognition RNN
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]  # latent_dim=4
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            func.reset_evals()

            end = time.time()
            # pred_z = odeint(func, z0, samp_ts, method=args.method).permute(1, 0, 2)
            pred_z, err = odeint_err(func, z0, samp_ts, method=args.method)
            batch_time_meter.update(time.time() - end)

            kin_states = quadratic_cost(pred_z)
            pred_z = pred_z.permute(1, 0, 2)
            pred_x = dec(pred_z)  # (1000, 100, 2)
            import pdb
            pdb.set_trace()

            # nfe
            epoch_nfe.append(func.num_evals())

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)

            # l1, l2 regularization
            l1 = torch.tensor([0.0], requires_grad=True).to(device)
            l2 = torch.tensor([0.0], requires_grad=True).to(device)
            for parameter in func.parameters():
                l1 = l1 + parameter.norm(1)
                l2 = l2 + parameter.norm(2)
            l1 = nn.Parameter(l1)
            l2 = nn.Parameter(l2)
            # loss += args.l1 * l1
            # loss += args.l2 * l2

            # dopri error term regularization
            # loss += args.dopri_lambda / torch.mean(torch.stack(err))  # 1/mean(step)
            loss += args.dopri_lambda * torch.mean(1/torch.stack(err))  # mean(1/step)

            # kinetic energy regularization
            loss += args.kinetic_lambda * torch.mean(kin_states)
            
            loss.backward()

            # for index, weight in enumerate(params, start=1):
            #     gradient, *_ = weight.grad.data
            #     print("Gradient of w{} w.r.t to L: ".format(index), gradient)
            # print("Gradient of l1 regularizer w.r.t to loss: ", l1.grad.data)
            # print("Gradient of l2 regularizer w.r.t to loss: ", l2.grad.data)
            # logger.info("Gradient of Dopri error regularizer w.r.t to loss: ", tmp.grad.data)
            
            optimizer.step()
            loss_meter.update(loss.item())

            # compute RMSE
            criterion = nn.MSELoss()
            rmse = torch.sqrt(criterion(pred_x, samp_trajs))

            # print(torch.autograd.gradcheck(odeint, (z0, samp_ts)))

            logger.info('#Obs: {}, Iter: {}, Running avg elbo: {:.4f}, RMSE: {:.4f}, NFE: {}, Time: {:.3f} (avg {:.3f})'.format(
                args.nsample, itr, -loss_meter.avg, rmse, iter_nfe, batch_time_meter.val, batch_time_meter.avg))

            # save model
            if itr == 1:
                best_rmse['itr'] = itr
                best_rmse['running_avg_elbo'] = -loss_meter.avg
                best_rmse['rmse'] = rmse
                best_rmse['nfe'] = iter_nfe
                best_rmse['time'] = batch_time_meter.val
                best_rmse['avg_time'] = batch_time_meter.avg
                if args.train_dir is not None:
                    ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
                    torch.save({
                        'iter': itr,
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'orig_trajs': orig_trajs,
                        'samp_trajs': samp_trajs,
                        'orig_ts': orig_ts,
                        'samp_ts': samp_ts,
                    }, ckpt_path)
                    logger.info('Stored ckpt at {}'.format(ckpt_path))
            # update model
            elif rmse < best_rmse['rmse']:
                best_rmse['itr'] = itr
                best_rmse['running_avg_elbo'] = -loss_meter.avg
                best_rmse['rmse'] = rmse
                best_rmse['nfe'] = iter_nfe
                best_rmse['time'] = batch_time_meter.val
                best_rmse['avg_time'] = batch_time_meter.avg
                if args.train_dir is not None:
                    ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
                    torch.save({
                        'iter': itr,
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'orig_trajs': orig_trajs,
                        'samp_trajs': samp_trajs,
                        'orig_ts': orig_ts,
                        'samp_ts': samp_ts,
                    }, ckpt_path)
                    logger.info('Stored ckpt at {}'.format(ckpt_path))
        epoch_nfe = np.array(epoch_nfe)
        logger.info('avg training NFE: {:.4f}'.format(np.mean(epoch_nfe)))
        logger.info('==>Best validation==>')
        logger.info('Iter: {}, Running avg elbo: {:.4f}, RMSE: {:.4f}, NFE: {}, Time: {:.3f} (avg {:.3f})'.format(
            best_rmse['itr'], best_rmse['running_avg_elbo'], best_rmse['rmse'], best_rmse['nfe'], best_rmse['time'], best_rmse['avg_time']))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'iter': best_rmse['itr'],
                'func_state_dict': func.state_dict(),
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'orig_trajs': orig_trajs,
                'samp_trajs': samp_trajs,
                'orig_ts': orig_ts,
                'samp_ts': samp_ts,
            }, ckpt_path)
            logger.info('Stored ckpt at {}'.format(ckpt_path))
    logger.info('Training complete after {} iters.'.format(itr))

    ## Test(extrapolation)
    if args.test:
        logger.info('*' * 15 + 'start testing' + '*' * 15)
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            samp_trajs = checkpoint['samp_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']
            logger.info('Loaded {}th ckpt from {}'.format(checkpoint['iter'], ckpt_path))
        with torch.no_grad():
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):  # 100
                obs = samp_trajs[:, t, :]  # (1000, 2). t'th point of 1000 spirals
                out, h = rec.forward(obs, h)  # recognition RNN
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]  # latent_dim=4
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            test_trajs = torch.from_numpy(test_trajs).float().to(device)
            test_ts = torch.from_numpy(test_ts).float().to(device)

            # forward in time and solve ode for extrapolation
            end = time.time()
            zs_neg = odeint(func, z0, test_ts, method=args.method).permute(1, 0, 2)
            tim = time.time() - end
            xs_neg = torch.flip(dec(zs_neg), dims=[0])

            # compute RMSE
            criterion = nn.MSELoss()
            rmse = torch.sqrt(criterion(xs_neg, test_trajs))
        logger.info('#Obs: {}, Test RMSE: {:.4f}, Time: {:.3f}'.format(args.nsample, rmse, tim))

    ## Visualize
    if args.visualize:
        with torch.no_grad():
            # sample from trajectorys' approx. posterior
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):  # 100
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            orig_ts = torch.from_numpy(orig_ts).float().to(device)

            # take first trajectory for visualization
            z0 = z0[0]  # either cc or cw ground-truth

            # ts_pos = np.linspace(0., 2. * np.pi, num=2000)  # t>0, reconstruction(interpolation)
            ts_pos = np.linspace(start, stop/2, num=2000)  # t>0, reconstruction(interpolation)
            # ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()  # t<0, extrapolation
            ts_neg = np.linspace(-stop/3, start, num=2000)[::-1].copy()  # t<0, extrapolation
            ts_pos = torch.from_numpy(ts_pos).float().to(device)
            ts_neg = torch.from_numpy(ts_neg).float().to(device)

            zs_pos = odeint(func, z0, ts_pos, method=args.method)
            zs_neg = odeint(func, z0, ts_neg, method=args.method)

            xs_pos = dec(zs_pos)
            xs_neg = torch.flip(dec(zs_neg), dims=[0])

        xs_pos = xs_pos.cpu().numpy()
        xs_neg = xs_neg.cpu().numpy()
        orig_traj = orig_trajs[0].cpu().numpy()
        samp_traj = samp_trajs[0].cpu().numpy()

        plt.figure()
        plt.plot(orig_traj[:, 0], orig_traj[:, 1],
                 'g', label='true trajectory')
        plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'c',
                 label='learned trajectory (t>0): interpolation')
        plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'r',
                 label='learned trajectory (t<0): extrapolation')
        plt.scatter(samp_traj[:, 0], samp_traj[:, 1], 
                    label='sampled data', s=3)  # observed points from gound-truth
        plt.legend()
        plt.savefig(args.save + '/vis.png', dpi=500)
        logger.info('Saved visualization figure at {}'.format(args.save + '/vis.png'))

    # remove checkpoint
    if args.train_dir is not None:
        logger.info('Removing checkpoint at {}'.format(args.train_dir + '/ckpt.pth'))
        for f in glob.glob(args.train_dir + '/ckpt.pth'):
            os.remove(f)
