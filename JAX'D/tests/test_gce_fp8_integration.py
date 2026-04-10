from pathlib import Path

import pytest


def test_deploy_profiler_script_exists_and_is_executable():
    script = Path("scripts/deploy_profiler_gce.sh")
    assert script.exists()
    assert script.stat().st_mode & 0o111


def test_fp8_zero3_hessian_callback_smoke():
    pl = pytest.importorskip("pytorch_lightning")
    torch = pytest.importorskip("torch")

    from src.callbacks.fp8_zero3_hessian_pl import FP8Zero3HessianPLCallback

    class DummyModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(3, 1)

        def training_step(self, batch, batch_idx):
            del batch_idx
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

    model = DummyModule()
    cb = FP8Zero3HessianPLCallback(
        monitor_every=1,
        warmup_steps=0,
        pl_tol=1e-6,
        use_checkpointing=False,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=4,
        callbacks=[cb],
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
        enable_model_summary=False,
    )

    loader = torch.utils.data.DataLoader(ToyDataset(), batch_size=1)
    trainer.fit(model, train_dataloaders=loader)

    assert cb.mu_global is not None
    assert cb.L_global is not None
    assert cb.L_global >= cb.mu_global
