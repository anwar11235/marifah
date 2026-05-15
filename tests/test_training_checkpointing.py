"""Unit tests for checkpointing: save/load round-trip and warm-start."""

import os
import tempfile
import torch
import torch.nn as nn
import pytest

from marifah.training.checkpointing import save_checkpoint, load_checkpoint, load_warmstart


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class TestCheckpointing:
    def test_round_trip(self, tmp_path):
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        path = str(tmp_path / "ckpt.pt")

        # Mutate weights
        with torch.no_grad():
            model.linear.weight.fill_(1.23)

        save_checkpoint(path, model, optimizer, None, step=10, epoch=2,
                        config_dict={"lr": 1e-3}, extra_metadata={"tag": "test"})

        model2 = TinyModel()
        meta = load_checkpoint(path, model2, optimizer=None)

        assert torch.allclose(model.linear.weight, model2.linear.weight), \
            "Weights not restored after round-trip"
        assert meta["step"] == 10
        assert meta["epoch"] == 2
        assert meta["extra_metadata"]["tag"] == "test"

    def test_atomic_write_on_interrupt(self, tmp_path):
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        path = str(tmp_path / "ckpt.pt")

        # First save succeeds
        save_checkpoint(path, model, optimizer, None, step=5, epoch=1, config_dict={})
        assert os.path.exists(path)

        # Verify no .tmp files left
        tmp_files = list(tmp_path.glob("*.pt.tmp"))
        assert len(tmp_files) == 0, f"Tmp files left behind: {tmp_files}"

    def test_optimizer_state_restored(self, tmp_path):
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        # Step optimizer once so it has non-trivial state
        out = model(torch.randn(2, 4))
        out.sum().backward()
        opt.step()
        opt.zero_grad()

        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model, opt, None, step=1, epoch=0, config_dict={})

        model2 = TinyModel()
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        load_checkpoint(path, model2, optimizer=opt2)

        # Compare optimizer state (step count)
        state1 = opt.state_dict()["state"]
        state2 = opt2.state_dict()["state"]
        for pid in state1:
            if "step" in state1[pid]:
                assert state1[pid]["step"] == state2[pid]["step"]

    def test_warmstart_fresh_optimizer(self, tmp_path):
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        with torch.no_grad():
            model.linear.weight.fill_(9.9)
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model, opt, None, step=100, epoch=10, config_dict={})

        model2 = TinyModel()
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)
        load_warmstart(path, model2)

        # Weights loaded
        assert torch.allclose(model.linear.weight, model2.linear.weight)
        # Optimizer not touched (fresh state)
        assert opt2.state_dict()["state"] == {}

    def test_load_model_only_no_optimizer(self, tmp_path):
        """load_checkpoint with optimizer=None should not raise."""
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, model, opt, None, step=3, epoch=0, config_dict={})

        model2 = TinyModel()
        meta = load_checkpoint(path, model2)  # no optimizer arg
        assert meta["step"] == 3
