import numpy as np
import pytest

from src.regge_bootstrap import ReggeExtendedBootstrap


def test_regge_fakeon_virtualization():
    solver = ReggeExtendedBootstrap(N_s=64, N_l=4, alpha=0.05, M2=2.4e23)
    s_mock = np.ones((solver.N_l, len(solver.s_grid)), dtype=complex) * 0.9
    delta_mock = np.zeros_like(s_mock)

    res = solver.run_full_regge_analysis(s_mock, delta_mock)
    assert res["fakeon_virtualized"]
    assert res["Re_alpha_at_M2"] < 0


def test_lightning_hessian_pl_callback():
    pl = pytest.importorskip("pytorch_lightning")
    torch = pytest.importorskip("torch")

    from src.callbacks.hessian_pl_callback import HessianPLCallback

    class DummyPPO(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(3, 1)

        def training_step(self, batch, batch_idx):
            x, _ = batch
            out = self.net(x)
            return (out**2).mean()

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    class ToyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            del idx
            return torch.randn(3), torch.tensor([0.0])

    model = DummyPPO()
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=4,
        callbacks=[HessianPLCallback(monitor_every_n_steps=1, pl_tol=1e-6, warmup_steps=0)],
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
        enable_model_summary=False,
    )

    loader = torch.utils.data.DataLoader(ToyDataset(), batch_size=1)
    trainer.fit(model, train_dataloaders=loader)

    cb = trainer.callbacks[0]
    assert cb.mu_est is not None
    assert cb.L_est is not None
    assert cb.L_est >= cb.mu_est
