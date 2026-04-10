import pytest


def test_memory_snapshot_shard_tracker_rotates_files(tmp_path):
    from src.regge_shard_map_memory_snapshot import MemorySnapshotShardTracker

    tracker = MemorySnapshotShardTracker(log_dir=str(tmp_path), run_id="r1", max_snapshots=2)
    for step in [1, 2, 3]:
        (tmp_path / "r1" / "profiles" / f"mem_step_{step}.pb.gz").write_text("x")

    tracker._cleanup_old_snapshots()

    files = sorted((tmp_path / "r1" / "profiles").glob("mem_step_*.pb.gz"))
    assert [f.name for f in files] == ["mem_step_2.pb.gz", "mem_step_3.pb.gz"]
    assert tracker.get_tensorboard_command().startswith("tensorboard --logdir=")


def test_cpu_fallback_zero_infinity_callback_smoke():
    pl = pytest.importorskip("pytorch_lightning")
    torch = pytest.importorskip("torch")

    from src.callbacks.zeroinfinity_cpu_fallback_pl import CPUFallbackZeroInfinityCallback

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
    cb = CPUFallbackZeroInfinityCallback(
        monitor_every=1,
        warmup_steps=0,
        pl_tol=1e-6,
        use_checkpointing=False,
        nvme_path="/path/that/does/not/exist",
        min_nvme_free_gb=0.0,
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

    assert cb.fallback_active is True
    assert cb.mu_global is not None
    assert cb.L_global is not None
    assert cb.L_global >= cb.mu_global
