import os
import random
from typing import Sequence, Dict, Tuple, Any, Union, Mapping, Optional
from tqdm.autonotebook import tqdm

from src.model import Expert, Discriminator, Discriminator
from src.arguments import Args
from src.utils import plot_scores_heatmap
from src.losses import ChamferDistanceLoss

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torch_geometric.data import DataLoader, Batch

class Trainer():
    def __init__(self, hparams: Args, expert: Expert, discriminator: Discriminator, train_dataset, test_dataset = None):
        """
        :param hparams: dictionary that contains all the hyperparameters
        :param experts:
        :param discriminator:
        :param train_dataset:
        :param test_dataset:
        """
        super().__init__()
        self.hparams = hparams
        
        # Experts
        self.expert = expert.to(self.hparams.device)

        # Discriminators
        self.discriminator = discriminator.to(self.hparams.device)

        # Optimizers
        self.E_optimizer, self.D_optimizer = self._configure_optimizers()

        # Datasets
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # Dataset for visualization of each digit
        self.perclass_dataset = {cls: set() for cls in range(self.train_dataset.num_classes)}
        for data in self.train_dataset:
            if data.y_transf.item() in self.perclass_dataset.keys():
                cls = data.y_transf.item()
                self.perclass_dataset[cls].add(data)

        # Dataloader
        self.train_loader, self.eval_loader, self.test_loader = self._prepare_loader()

        # Losses
        self.chamfer = ChamferDistanceLoss()
        self.chamfer.to(self.hparams.device)
        self.bce = torch.nn.BCELoss()
        self.bce.to(self.hparams.device)

        # Restore from last save
        self._current_epoch = 1
        self._current_step = 0
        self.loss = {}

    def _prepare_loader(self) -> Tuple[DataLoader, DataLoader]:
        """
        Create dataloaders.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.n_cpu, follow_batch=['pos', 'pos_transf'])
        eval_loader = DataLoader(self.train_dataset, batch_size=2 * self.hparams.batch_size, shuffle=True, num_workers=self.hparams.n_cpu, follow_batch=['pos', 'pos_transf'])
        if self.test_dataset != None:
            test_loader = DataLoader(self.test_dataset, batch_size=self.hparams.batch_size * 2, shuffle=False, num_workers=self.hparams.n_cpu, follow_batch=['pos', 'pos_transf'])
        else:
            test_loader = None
        return train_loader, eval_loader, test_loader

    def _configure_optimizers(self) -> Sequence[optim.Optimizer]:
        """
        Instantiate the optimizers; each expert has an optimizer and the discriminator one is append as last element.

        :returns: the optimizers
        """
        # Experts optimizers
        E_optimizer = optim.Adam(self.expert.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b1))
        # Discriminator optimizer
        D_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b1))
        return E_optimizer, D_optimizer

    def initialize_expert(self) -> None:
        """
            Initialization of the expert as identity.

            :param expert:
            :param i:
            :param optimizer:
        """
        optimizer = optim.Adam(self.expert.parameters(), lr=self.hparams.lr)
        print(f'Initializing expert as identity on perturbed data')
        self.expert.train()
        for epoch in range(1, self.hparams.epochs_init + 1):
            n_samples = 0
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}', bar_format='{desc} {percentage:0.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  {postfix}')
            for data in progress_bar:
                batch_size = data.num_graphs
                n_samples += batch_size
                data = data.to(self.hparams.device)
                optimizer.zero_grad()
                out = self.expert(pos=data.pos_transf, batch=data.pos_transf_batch)
                loss = self.chamfer(out.view(batch_size, -1, self.hparams.in_channels), data.pos_transf.view(batch_size, -1, self.hparams.in_channels))
                loss.backward()
                total_loss += loss.item() * batch_size
                optimizer.step()
                progress_bar.set_description(f'Epoch {epoch}')
                progress_bar.set_postfix({'loss': total_loss/n_samples})

        # save initialization                       
        init = {
            'model': self.expert.state_dict(),
        }
        if not os.path.isdir(os.path.join(self.hparams.models_path, 'init')):
            os.mkdir(os.path.join(self.hparams.models_path, 'init'))
        torch.save(init, os.path.join(self.hparams.models_path, 'init', 'expert.pth'))

    def train(self) -> None:
        """
        Train the model in adversarial fashion.

        :param initialize:
        """
        config = {
            'batch_size': self.hparams.batch_size,
            'learning_rate': self.hparams.lr,
            'num_points': self.hparams.num_points,
            'dataset': "pointcloud mnist",
            'optimizer': 'adam',
        }
        wandb.init(id=self.hparams.id_wandb, project="causal-mechanisms-on-pointclouds", config=config, resume=self.hparams.resume)
        models = (self.discriminator, self.expert)
        wandb.watch(models, log="all")

        # if wandb.run.resumed and os.path.isfile(f'{self.hparams.models_path}/ckpt.pth'):
        #     # restore the model
        #     self.load_checkpoint()

        # train the models
        for epoch in range(self._current_epoch, self.hparams.epochs + 1):
            self._training_loop(epoch)
            
            score = self.evaluation()
            wandb.log({f'{self.train_dataset.idx_to_transf[0]}_scores': score, 'epoch': epoch})

            # save checkpoint
            self.save_checkpoint(epoch)

            if epoch % 10 == 0:
                self.visualization()

        wandb.finish()

    def _training_loop(self, epoch):
        """
        Training loop.

        :param epoch:
        """
        self.discriminator.train()
        self.expert.train()

        # Keep track of losses
        losses = {
            'D_batch_loss': [],
            'E_batch_loss': [],
        }

        # Iterate through data
        progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader), bar_format='{desc}: {percentage:0.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  {postfix}', ncols=500)
        for idx, data in progress_bar:
            self._current_step += 1
            self._training_step(data, losses)

        D_loss = sum(losses['D_batch_loss']) / len(losses['D_batch_loss'])
        E_loss = sum(losses['E_batch_loss']) / len(losses['E_batch_loss'])
        self.loss = {'epoch': epoch, 'D_loss': D_loss, 'E_loss': E_loss}
        wandb.log(self.loss)

    def _training_step(self, data, losses):
        """
        Implements a single training step.

        :param data: current training batch
        :param losses: dictionary of the losses
        """

        data = data.to(self.hparams.device)
        x_canon, x_transf = data.pos, data.pos_transf
        batch_transf = data.pos_transf_batch

        batch_size = data.num_graphs

        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(self.hparams.device)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(self.hparams.device)

        # Train expert
        self.E_optimizer.zero_grad()

        fake_point = self.expert(pos=x_transf, batch=batch_transf)

        G_fake = self.discriminator(pos=fake_point.view(batch_size, -1, 3))
        g_loss = self.bce(G_fake, valid)

        losses['E_batch_loss'].append(g_loss.item())

        g_loss.backward()
        self.E_optimizer.step()

        # Train discriminator
        self.D_optimizer.zero_grad()

        # Real points
        D_real = self.discriminator(pos=x_canon.view(batch_size, -1, 3))
        D_real_loss = self.bce(D_real, valid)

        # Fake points
        D_fake = self.discriminator(pos=fake_point.detach().view(batch_size, -1, 3))
        D_fake_loss = self.bce(D_fake, fake)
        
        # Discriminator loss
        d_loss = (D_real_loss + D_fake_loss) / 2

        losses['D_batch_loss'].append(d_loss.item())

        d_loss.backward()
        self.D_optimizer.step()

        batch_losses = {'batch': self._current_step, 'D_batch_loss': d_loss.item(), 'E_batch_loss': g_loss.item()}
        wandb.log(batch_losses)

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save the checkpoint of the model and losses at current epoch.
        """
        ckpt = {
                'epoch': epoch,
                'D_state_dict': self.discriminator.state_dict(),
                'D_optim': self.D_optimizer.state_dict(),
                'D_loss': self.loss['D_loss'],
                'E_state_dict':self.expert.state_dict(),
                'E_loss': self.loss[f'E_loss'],
                'E_optim': self.E_optimizer.state_dict(),
        }
        
        torch.save(ckpt, f'{self.hparams.models_path}/ckpt.pth')

    def load_checkpoint(self) -> None:
        """
        Load the models of a given epoch.

        :param epoch:
        """
        print('Loading checkpoint...')
        ckpt = torch.load(f'{self.hparams.models_path}/ckpt.pth', map_location=self.hparams.device)
        self._current_epoch = ckpt['epoch'] + 1
        self._current_step = ckpt['epoch'] * len(self.train_loader)
        self.loss['D_loss'] = ckpt['D_loss']
        self.discriminator.load_state_dict(ckpt['D_state_dict'])
        self.D_optimizer.load_state_dict(ckpt['D_optim'])
        self.expert.load_state_dict(ckpt[f'E_state_dict'])
        self.loss[f'E_loss'] = ckpt[f'E_loss']
        self.E_optimizer.load_state_dict(ckpt[f'E_optim'])

    @torch.no_grad()
    def evaluation(self) -> torch.Tensor:
        """
        TODO
        """
        with torch.no_grad():
            self.discriminator.eval()
            self.expert.eval()

            for batch in tqdm(self.eval_loader, desc="Eval... ", total=len(self.eval_loader)):
                batch = batch.to(self.hparams.device)
                batch_size = batch.num_graphs
                x_transf = batch.pos_transf
                batch_transf = batch.pos_transf_batch

                # Pass transformed data through experts
                exp_output = self.expert(pos=x_transf, batch=batch_transf)
                # Get the score for each pointclopud in the batch
                exp_scores = self.discriminator(exp_output.view(batch_size, -1, 3))

            # return the mean of the scores
            score = exp_scores.mean()
            return score

    @torch.no_grad()
    def test(self) -> None:
        """
        Test the model.
        """
        assert self.test_dataset != None and self.test_loader != None
        transf_scores = {idx_transf: torch.zeros(self.hparams.num_experts, device=self.hparams.device) for idx_transf in self.test_dataset.idx_to_transf.keys()}
        transf_n_samples = {idx_transf: 0 for idx_transf in transf_scores.keys()}

        self.discriminator.eval()
        for _, batch in tqdm(enumerate(self.test_loader), desc=f'Testing the model...', total=len(self.test_loader), bar_format='{desc} {percentage:0.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            batch = batch.to(self.hparams.device)
            batch_size = batch.num_graphs
            x_transf = batch.pos_transf
            batch_transf = batch.pos_transf_batch

            # Pass transformed data through experts
            experts_scores = []
            for expert in self.experts:
                expert.eval()
                exp_output = expert(pos=x_transf, batch=batch_transf)
                exp_scores = self.discriminator(exp_output.view(batch_size, -1, 3))
                experts_scores.append(exp_scores)
            
            experts_scores = torch.cat(experts_scores, dim=1)

            for idx in self.test_dataset.idx_to_transf.keys():
                indices = batch.transf.eq(torch.Tensor([idx]).to(self.hparams.device)).nonzero(as_tuple=False).squeeze(dim=-1)
                n_indices = indices.shape[0]
                if n_indices > 0:
                    transf_n_samples[idx] += n_indices
                    transf_scores[idx] += experts_scores[indices].sum(dim=0)

        for idx in transf_scores.keys():
            transf_scores[idx] = transf_scores[idx] / transf_n_samples[idx]

        plot_scores_heatmap(transf_scores, self.test_dataset.idx_to_transf, self.hparams.project_dir)

    @torch.no_grad()
    def visualization(self) -> None:
        point_clouds = {}
        with torch.no_grad():
            self.discriminator.eval()
            self.expert.eval()

            # sample a point cloud for each digit from the dataset
            samples = [random.sample(self.perclass_dataset[cls], 1)[0] for cls in range(self.train_dataset.num_classes)]
            for data in samples:
                digit = data.y_transf.item()
                batch = Batch.from_data_list([data], follow_batch=['pos', 'pos_transf']).to(self.hparams.device)
                pred_pointcloud = self.expert(batch.pos_transf, batch.pos_transf_batch)
                score = self.discriminator(pred_pointcloud.unsqueeze(0)).item()
                point_clouds[f'pointcloud {digit}'] = wandb.Object3D(data.pos_transf.cpu().numpy())
                point_clouds[f'pointcloud {digit} - score: {score}'] = wandb.Object3D(pred_pointcloud.cpu().numpy())
        wandb.log(point_clouds)
