import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to


class StackedHeads(nn.Module):
    """
    Environment-specific linear heads stored as a single linear layer.
    The heads are stacked row-wise.
    weight: (num_heads * output_dim, feature_dim)
    bias: (num_heads * output_dim)
    Rows [h * output_dim : (h+1) * output_dim] hold head h.
    """
    def __init__(self, heads):
        super().__init__()
        self.n_heads = len(heads)
        self.d_out = heads[0].out_features
        self.weight = nn.Parameter(torch.cat([h.weight.data for h in heads], dim=0))
        self.bias = nn.Parameter(torch.cat([h.bias.data for h in heads], dim=0))

    def forward(self, features):
        # All expert predictions in one matrix multiplication: (batch_size, num_heads, output_dim)
        return F.linear(features, self.weight, self.bias).view(-1, self.n_heads, self.d_out)

    def centroid(self):
        # the centroid head, used at deployment
        w_bar = self.weight.view(self.n_heads, self.d_out, -1).mean(dim=0)
        b_bar = self.bias.view(self.n_heads, self.d_out).mean(dim=0)
        return w_bar, b_bar


class REC(SingleModelAlgorithm):
    """Random-Effects Centroid: expert loss + lambda * centroid loss, with lambda > 0."""
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train):
        assert config.uniform_over_groups
        assert config.lambda_avg > 0, 'REC requires lambda_avg > 0; use ExpertOnly for lambda = 0'
        featurizer = initialize_model(config, d_out, is_featurizer=True)[0]
        n_heads = int(is_group_in_train.sum())
        heads = StackedHeads([nn.Linear(featurizer.d_out, d_out) for _ in range(n_heads)])
        model = nn.Sequential(featurizer, heads)
        super().__init__(config=config, model=model, grouper=grouper, loss=loss, metric=metric, n_train_steps=n_train_steps)

        # -1 for groups without a head
        train_groups = torch.where(is_group_in_train)[0]
        self.head_to_group = {idx: group.item() for idx, group in enumerate(train_groups)}
        self.group_to_head = torch.full((len(is_group_in_train),), -1, dtype=torch.long, device=config.device)
        self.group_to_head[train_groups.to(config.device)] = torch.arange(n_heads, device=config.device)

        self.n_heads = n_heads
        self.lambda_avg = config.lambda_avg
        self.logged_fields += ['expert_loss', 'centroid_loss']

    def process_batch(self, batch, unlabeled_batch=None):
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        self.features = self.model[0](x)

        # prediction of the deployed model
        w_bar, b_bar = self.model[1].centroid()
        outputs = F.linear(self.features, w_bar, b_bar)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        return results

    def objective(self, results):
        g = results['g']
        y_true = results['y_true']

        if not self.is_training:
            centroid_loss = self.loss.compute(results['y_pred'], y_true, return_dict=False)
            self.save_metric_for_logging(results, 'expert_loss', 0.0)
            self.save_metric_for_logging(results, 'centroid_loss', centroid_loss)
            return centroid_loss

        head_idx = self.group_to_head[g]
        valid = head_idx >= 0
        head_idx, y, feats = head_idx[valid], y_true[valid], self.features[valid]
        y_pred_cent = results['y_pred'][valid]

        # weight per sample = 1 / (number of environments in the batch * number of samples from its environment)
        counts = torch.bincount(head_idx, minlength=self.n_heads)
        n_envs = (counts > 0).sum()
        weights = 1.0 / (counts[head_idx] * n_envs)

        # expert loss
        all_preds = self.model[1](feats) # (batch_size, num_heads, output_dim)
        expert_pred = all_preds[torch.arange(len(head_idx), device=self.device), head_idx]
        expert_loss = (weights * self.loss.compute_element_wise(expert_pred, y, return_dict=False)).sum()

        # centroid loss
        centroid_loss = (weights * self.loss.compute_element_wise(y_pred_cent, y, return_dict=False)).sum()

        total_loss = expert_loss + self.lambda_avg * centroid_loss

        self.save_metric_for_logging(results, 'expert_loss', expert_loss)
        self.save_metric_for_logging(results, 'centroid_loss', centroid_loss)
        return total_loss


class ExpertOnly(SingleModelAlgorithm):
    """
    Ablation of REC without the centroid loss (lambda = 0).
    Each head is trained only on its own environment, and the centroid head is deployed.
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train):
        assert config.uniform_over_groups
        featurizer = initialize_model(config, d_out, is_featurizer=True)[0]
        model = nn.Sequential(featurizer, nn.ModuleList([nn.Linear(featurizer.d_out, d_out) for _ in range(is_group_in_train.sum())]))
        super().__init__(config=config, model=model, grouper=grouper, loss=loss, metric=metric, n_train_steps=n_train_steps)

        self.head_to_group = {idx: group.item() for idx, group in enumerate(torch.where(is_group_in_train)[0])}

    def process_batch(self, batch, unlabeled_batch=None):
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        self.features = self.model[0](x)

        # prediction of the deployed model
        avg_weight = torch.stack([c.weight for c in self.model[1]]).mean(dim=0)
        avg_bias = torch.stack([c.bias for c in self.model[1]]).mean(dim=0)
        outputs = F.linear(self.features, avg_weight, avg_bias)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        return results

    def objective(self, results):
        g = results['g']
        y_true = results['y_true']

        # expert loss
        expert_losses = []
        for head_idx, group_idx in self.head_to_group.items():
            mask = (g == group_idx)
            if mask.sum() > 0:
                head_output = self.model[1][head_idx](self.features[mask])
                head_loss = self.loss.compute(head_output, y_true[mask], return_dict=False)
                expert_losses.append(head_loss)
        if len(expert_losses) > 0:
            expert_loss = torch.stack(expert_losses).mean()
        else:
            expert_loss = torch.tensor(0.0, device=self.device)

        return expert_loss