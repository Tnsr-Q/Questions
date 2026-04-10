import pytest


def test_profiled_shard_map_regge_solver_smoke(tmp_path):
    jnp = pytest.importorskip("jax.numpy")

    from src.regge_shard_map_profiler import ProfiledShardedReggeSolver

    profile_dir = tmp_path / "profiles"
    solver = ProfiledShardedReggeSolver(N_t=16, t_min=1e-2, t_max=1e2, profile_dir=str(profile_dir))
    delta_mock = jnp.zeros(16) + 0.05

    traj, meta = solver.scan_with_profiler(delta_mock, run_id="test_run")
    cert = solver.verify_fakeon_virtualization(traj)

    assert traj.shape == (16,)
    assert meta["status"] == "PROFILING_COMPLETE"
    assert meta["run_id"] == "test_run"
    assert meta["devices"] >= 1
    assert cert["status"] in {"VERIFIED", "PENDING"}


def test_compressed_zero3_hessian_callback_smoke():
    pl = pytest.importorskip("pytorch_lightning")
    torch = pytest.importorskip("torch")

    from src.callbacks.zero3_compressed_hessian_pl import CompressedZero3HessianPLCallback

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
    cb = CompressedZero3HessianPLCallback(
        monitor_every=1,
        warmup_steps=0,
        pl_tol=1e-6,
        use_checkpointing=False,
        compress_gradients=False,
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
