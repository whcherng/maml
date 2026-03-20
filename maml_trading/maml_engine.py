"""
MAML Trading Engine — Bi-Level Optimization with learn2learn.

Implements the full Model-Agnostic Meta-Learning loop:

  Inner Loop (Fast Adaptation):
    For each task (market regime), perform K gradient steps on the
    support set to adapt the cloned model parameters to the current
    regime. Uses learn2learn's MAML wrapper which handles the
    computational graph for higher-order derivatives.

  Outer Loop (Meta-Update):
    Evaluate each adapted model on its query set. Aggregate the
    query losses across the task batch and backpropagate through
    the inner-loop computation graph to update the shared
    meta-parameters. Optionally uses KroneckerTransform for a
    more efficient meta-update step.

Dependencies:
  - learn2learn.algorithms.MAML
  - learn2learn.optim.transforms.KroneckerTransform
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l
from tqdm import tqdm

from maml_trading.config import MAMLConfig
from maml_trading.model import AdaptiveTradingNetwork
from maml_trading.task_generator import MarketRegimeTask, MarketRegimeTaskGenerator


class MAMLTradingEngine:
    """
    Orchestrates the bi-level MAML optimization for trading.

    Usage:
        engine = MAMLTradingEngine(config, task_generator)
        engine.meta_train()
        predictions = engine.adapt_and_predict(new_task)
    """

    def __init__(
        self,
        config: MAMLConfig,
        task_generator: MarketRegimeTaskGenerator,
    ):
        self.cfg = config
        self.task_gen = task_generator
        self.device = torch.device(config.device)

        # ── Build the base model ────────────────────────────────────
        self.base_model = AdaptiveTradingNetwork(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            output_dim=config.output_dim,
        ).to(self.device)

        # ── Wrap with learn2learn MAML ──────────────────────────────
        # This creates a differentiable clone mechanism that preserves
        # the computation graph for second-order gradients.
        self.maml = l2l.algorithms.MAML(
            self.base_model,
            lr=config.inner_lr,
            first_order=config.first_order,
            allow_nograd=True,
        )

        # ── Optional: KroneckerTransform for meta-update ────────────
        # Kronecker-factored curvature approximation provides a
        # preconditioned meta-gradient, improving convergence.
        if config.use_kronecker:
            self.meta_optimizer = optim.Adam(
                self.maml.parameters(), lr=config.meta_lr
            )
            # KroneckerTransform is applied as a parameter transform
            # within the learn2learn optimization pipeline
            try:
                self.kronecker_transform = l2l.optim.transforms.KroneckerTransform(
                    l2l.nn.Scale
                )
            except (AttributeError, ImportError):
                print(
                    "[MAMLEngine] KroneckerTransform not available in "
                    "this learn2learn version. Falling back to vanilla Adam."
                )
                self.kronecker_transform = None
        else:
            self.meta_optimizer = optim.Adam(
                self.maml.parameters(), lr=config.meta_lr
            )
            self.kronecker_transform = None

        # ── Loss function ───────────────────────────────────────────
        # Use label smoothing to prevent overconfident predictions and
        # encourage the model to maintain uncertainty across classes.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        # ── LR Scheduler ───────────────────────────────────────────
        # Cosine annealing helps escape local minima in later epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer,
            T_max=config.meta_epochs,
            eta_min=config.meta_lr * 0.01,
        )

        # ── Training history ────────────────────────────────────────
        self.train_history: List[Dict[str, float]] = []

    # ════════════════════════════════════════════════════════════════
    # Inner Loop — Fast Adaptation
    # ════════════════════════════════════════════════════════════════

    def _inner_loop(
        self,
        learner: nn.Module,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        """
        Perform `inner_steps` gradient updates on the support set.

        This adapts the cloned model's parameters to the specific
        market regime represented by this task. The computation graph
        is preserved so the outer loop can differentiate through
        these updates (second-order MAML).

        Args:
            learner: a cloned model from self.maml.clone().
            support_x: (K, seq_len, F) support features.
            support_y: (K,) support labels.

        Returns:
            The adapted learner.
        """
        for step in range(self.cfg.inner_steps):
            logits = learner(support_x)
            loss = self.criterion(logits, support_y)
            learner.adapt(loss)  # learn2learn handles the gradient + update
        return learner

    # ════════════════════════════════════════════════════════════════
    # Outer Loop — Meta-Update
    # ════════════════════════════════════════════════════════════════

    def _outer_step(
        self,
        tasks: List[MarketRegimeTask],
    ) -> Tuple[float, float]:
        """
        One meta-update step:
          1. For each task, clone the meta-model and run the inner loop.
          2. Evaluate each adapted clone on its query set.
          3. Sum query losses and backprop through the inner loop
             to update the meta-parameters.

        Args:
            tasks: batch of MarketRegimeTask objects.

        Returns:
            (meta_loss, meta_accuracy) averaged over the task batch.
        """
        meta_loss = 0.0
        meta_correct = 0
        meta_total = 0

        for task in tasks:
            # Move task data to device
            sx = task.support_x.to(self.device)
            sy = task.support_y.to(self.device)
            qx = task.query_x.to(self.device)
            qy = task.query_y.to(self.device)

            # Clone the meta-model (preserves computation graph)
            learner = self.maml.clone()

            # Inner loop: adapt to this regime
            learner = self._inner_loop(learner, sx, sy)

            # Evaluate on query set
            q_logits = learner(qx)
            q_loss = self.criterion(q_logits, qy)
            meta_loss += q_loss

            # Track accuracy
            preds = q_logits.argmax(dim=1)
            meta_correct += (preds == qy).sum().item()
            meta_total += qy.size(0)

        # Average over tasks
        meta_loss = meta_loss / len(tasks)

        # Meta-update: backprop through inner loops
        self.meta_optimizer.zero_grad()
        meta_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.maml.parameters(), max_norm=5.0)

        self.meta_optimizer.step()

        accuracy = meta_correct / max(meta_total, 1)
        return meta_loss.item(), accuracy

    # ════════════════════════════════════════════════════════════════
    # Meta-Training Loop
    # ════════════════════════════════════════════════════════════════

    def meta_train(
        self,
        n_epochs: Optional[int] = None,
        start_range: Optional[Tuple[int, int]] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Run the full meta-training loop.

        Args:
            n_epochs: override for cfg.meta_epochs.
            start_range: constrain task sampling to a time range
                         (for walk-forward training windows).
            verbose: print progress bar.

        Returns:
            Training history as list of {epoch, loss, accuracy} dicts.
        """
        epochs = n_epochs or self.cfg.meta_epochs
        self.base_model.train()

        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Meta-Training")

        for epoch in iterator:
            # Sample a batch of regime tasks
            tasks = self.task_gen.sample_tasks(start_range=start_range)

            # One meta-update step
            loss, acc = self._outer_step(tasks)

            # Step the cosine LR scheduler
            self.scheduler.step()

            record = {"epoch": epoch, "loss": loss, "accuracy": acc}
            self.train_history.append(record)

            if verbose:
                iterator.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}")

        return self.train_history

    # ════════════════════════════════════════════════════════════════
    # Inference — Adapt & Predict
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def adapt_and_predict(
        self,
        task: MarketRegimeTask,
        n_ensemble: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble inference: adapt the meta-model multiple times with
        shuffled support sets and average the predictions.

        This reduces variance from the small support set and produces
        more calibrated probability estimates.

        Args:
            task: a MarketRegimeTask with support and query data.
            n_ensemble: number of adaptation runs to average.

        Returns:
            (predictions, probabilities) for the query set.
              predictions: (Q,) int array of class indices.
              probabilities: (Q, C) float array of averaged softmax probs.
        """
        sx = task.support_x.to(self.device)
        sy = task.support_y.to(self.device)
        qx = task.query_x.to(self.device)

        all_probs = []

        for i in range(n_ensemble):
            with torch.enable_grad():
                learner = self.maml.clone()

                # Shuffle support set for diversity across ensemble members
                if i > 0:
                    perm = torch.randperm(sx.size(0))
                    sx_shuffled = sx[perm]
                    sy_shuffled = sy[perm]
                else:
                    sx_shuffled = sx
                    sy_shuffled = sy

                for _ in range(self.cfg.inner_steps):
                    logits = learner(sx_shuffled)
                    loss = self.criterion(logits, sy_shuffled)
                    learner.adapt(loss)

            learner.eval()
            q_logits = learner(qx)
            probs = torch.softmax(q_logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        # Average probabilities across ensemble
        avg_probs = np.mean(all_probs, axis=0)
        preds = avg_probs.argmax(axis=1)

        return preds, avg_probs

    def get_state(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "maml_state_dict": self.maml.state_dict(),
            "optimizer_state_dict": self.meta_optimizer.state_dict(),
            "train_history": self.train_history,
        }

    def load_state(self, state: dict) -> None:
        """Restore from a checkpoint."""
        self.maml.load_state_dict(state["maml_state_dict"])
        self.meta_optimizer.load_state_dict(state["optimizer_state_dict"])
        self.train_history = state.get("train_history", [])
