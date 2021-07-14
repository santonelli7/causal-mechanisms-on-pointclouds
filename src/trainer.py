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
    def __init__(self, hparams: Args, experts: Expert, discriminator: Discriminator, train_dataset, test_dataset = None):
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
        self.experts = [expert.to(self.hparams.device) for expert in experts]

        # Discriminators
        self.discriminator = discriminator.to(self.hparams.device)

        # Optimizers
        self.optimizers = self._configure_optimizers()

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
        self.bce = torch.nn.BCELoss()

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
        optimizers = [optim.Adam(expert.parameters(), lr=self.hparams.lr) for expert in self.experts]
        # Discriminator optimizer
        optimizers.append(optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr))
        assert len(optimizers) == len(self.experts) + 1
        return optimizers
    
    def initialize_experts(self) -> None:
        """
            Loop to initialize all experts.

            :param optimizers:
        """
        for i, (expert, optimizer) in enumerate(zip(self.experts, self.optimizers[:-1]), 1):
            self._initialize_expert(expert, i, optimizer)

            # save initialization                       
            init = {
                'model': expert.state_dict(),
                'optim': optimizer.state_dict()
            }
            torch.save(init, f'{self.hparams.models_path}/init/Expert_{i}_init.pth')

    def _initialize_expert(self, expert: Expert, i: int, optimizer: optim.Optimizer) -> None:
        """
            Initialization of the expert as identity.

            :param expert:
            :param i:
            :param optimizer:
        """
        print(f'Initializing expert {i} as identity on perturbed data')
        expert.train()
        for epoch in range(1, self.hparams.epochs_init + 1):
            n_samples = 0
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}', bar_format='{desc} {percentage:0.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  {postfix}')
            for data in progress_bar:
                batch_size = data.num_graphs
                n_samples += batch_size
                data = data.to(self.hparams.device)
                optimizer.zero_grad()
                out = expert(pos=data.pos_transf, batch=data.pos_transf_batch)
                loss = self.chamfer(out.view(batch_size, -1, self.hparams.in_channels), data.pos_transf.view(batch_size, -1, self.hparams.in_channels))
                loss.backward()
                total_loss += loss.item() * batch_size
                optimizer.step()
                progress_bar.set_description(f'Epoch {epoch}')
                progress_bar.set_postfix({'loss': total_loss/n_samples})

    def train(self, initialize: bool = False) -> None:
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
        models = tuple(self.experts)
        models += (self.discriminator,)
        wandb.watch(models, log="all")

        if wandb.run.resumed and os.path.isfile(f'{self.hparams.models_path}/ckpt.pth'):
            # restore the model
            self.load_checkpoint()

        # train the models
        for epoch in range(self._current_epoch, self.hparams.epochs + 1):
            self._training_loop(epoch)
            
            self.evaluation()

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
        for i, expert in enumerate(self.experts):
            expert.train()

        # Keep track of losses
        losses = {
            'D_batch_loss': [],
            'E_batch_loss': [[] for i in range(len(self.experts))],
        }

        # Iterate through data
        progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=len(self.train_loader), bar_format='{desc}: {percentage:0.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]  {postfix}', ncols=500)
        for idx, data in progress_bar:
            self._training_step(idx, data, losses)

        D_loss = sum(losses['D_batch_loss']) / len(losses['D_batch_loss'])
        self.loss = {'epoch': epoch, 'D_loss': D_loss}
        for i in range(len(self.experts)):
            E_loss = sum(losses['E_batch_loss'][i]) / len(losses['E_batch_loss'][i])
            self.loss[f'E_{i+1}_loss'] = E_loss
        wandb.log(self.loss)

    def _training_step(self, idx, data, losses):
        """
        Implements a single training step.

        :param data: current training batch
        :param losses: dictionary of the losses
        """
        self._current_step += idx

        data = data.to(self.hparams.device)
        x_canon, x_transf = data.pos, data.pos_transf
        batch_transf = data.pos_transf_batch

        batch_size = data.num_graphs

        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(self.hparams.device)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(self.hparams.device)

        self.optimizers[-1].zero_grad()

        # Real points
        D_real = self.discriminator(pos=x_canon.view(batch_size, -1, 3))
        D_real_loss = self.bce(D_real, valid)

        # Fake points
        D_fake_loss = 0
        exp_outputs = []
        exp_scores = []
        for i, expert in enumerate(self.experts):
            fake_point = expert(pos=x_transf, batch=batch_transf)
            D_fake = self.discriminator(pos=fake_point.detach().view(batch_size, -1, 3))
            D_fake_loss += self.bce(D_fake, fake)
            exp_outputs.append(fake_point.view(batch_size, 1, -1, self.hparams.in_channels))
            exp_scores.append(D_fake)
        D_fake_loss = D_fake_loss / self.hparams.num_experts

        # Discriminator loss
        d_loss = (D_real_loss + D_fake_loss) / 2

        d_loss.backward()
        self.optimizers[-1].step()

        losses['D_batch_loss'].append(d_loss.item())
        batch_losses = {'D_batch_loss': d_loss.item(), 'step': self._current_step}
        
        # Train experts
        exp_outputs = torch.cat(exp_outputs, dim=1)
        exp_scores = torch.cat(exp_scores, dim=1)
        mask_winners = exp_scores.argmax(dim=1)

        # Update each expert on samples it won
        for i, expert in enumerate(self.experts):
            
            self.optimizers[i].zero_grad()
            winning_indexes = mask_winners.eq(i).nonzero().squeeze(dim=-1)
            n_expert_samples = winning_indexes.size(0)
            if n_expert_samples > 0:
                winning_samples = exp_outputs[winning_indexes, i]
                G_fake = self.discriminator(pos=winning_samples)
                g_loss = self.bce(G_fake, valid)
                g_loss.backward()
                self.optimizers[i].step()

                losses['E_batch_loss'][i].append(g_loss.item())
                batch_losses[f'E_{i+1}_batch_loss'] = g_loss.item()
        wandb.log(batch_losses)

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save the checkpoint of the model and losses at current epoch.
        """
        ckpt = {
                'epoch': epoch,
                'D_state_dict': self.discriminator.state_dict(),
                'D_optim': self.optimizers[-1].state_dict(),
                'D_loss': self.loss['D_loss'],
                }
        for i, expert in enumerate(self.experts):
            ckpt[f'E_{i+1}_state_dict'] =  expert.state_dict()
            ckpt[f'E_{i+1}_loss'] = self.loss[f'E_{i+1}_loss']
            ckpt[f'E_{i+1}_optim'] = self.optimizers[i].state_dict()
        
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
        self.optimizers[-1].load_state_dict(ckpt['D_optim'])
        for i, expert in enumerate(self.experts):
            expert.load_state_dict(ckpt[f'E_{i+1}_state_dict'])
            self.loss[f'E_{i+1}_loss'] = ckpt[f'E_{i+1}_loss']
            self.optimizers[i].load_state_dict(ckpt[f'E_{i+1}_optim'])

    @torch.no_grad()
    def evaluation(self) -> torch.Tensor:
        """
        TODO
        """
        self.discriminator.eval()

        # transf_scores[i] contains the scores that experts get for their outputs giving as input transformed samples with the mechanism i
        transf_scores = [torch.zeros(self.hparams.num_experts, device=self.hparams.device) for idx_transf in self.train_dataset.idx_to_transf.keys()]
        transf_n_samples = [0 for _ in self.train_dataset.idx_to_transf.keys()]

        for batch in tqdm(self.eval_loader, desc="Eval... ", total=len(self.eval_loader)):
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

            for idx, _ in enumerate(transf_scores):
                indices = batch.transf.eq(torch.Tensor([idx]).to(self.hparams.device)).nonzero(as_tuple=False).squeeze(dim=-1)
                n_indices = indices.shape[0]
                if n_indices > 0:
                    transf_n_samples[idx] += n_indices
                    transf_scores[idx] += experts_scores[indices].sum(dim=0)

        assert len(transf_scores) == len(transf_n_samples)
        scores = {}
        for idx, score in enumerate(transf_scores):
            transf_scores[idx] = score / transf_n_samples[idx]
            scores[f'{self.train_dataset.idx_to_transf[idx]}_scores'] = transf_scores[idx]
        wandb.log(scores)

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
            for expert in self.experts:
                expert.eval()

                # sample a point cloud for each digit from the dataset
                samples = [random.sample(self.perclass_dataset[cls], 1)[0] for cls in range(self.train_dataset.num_classes)]
                for data in samples:
                    digit = data.y_transf.item()
                    batch = Batch.from_data_list([data], follow_batch=['pos', 'pos_transf']).to(self.hparams.device)
                    pred_pointcloud = expert(batch.pos_transf, batch.pos_transf_batch)
                    score = self.discriminator(pred_pointcloud.unsqueeze(0)).item()
                    point_clouds[f'pointcloud {digit}'] = wandb.Object3D(data.pos_transf.cpu().numpy())
                    point_clouds[f'pointcloud {digit} - score: {score}'] = wandb.Object3D(pred_pointcloud.cpu().numpy())
        wandb.log(point_clouds)
