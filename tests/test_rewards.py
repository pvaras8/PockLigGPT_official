import pytest

from pockliggpt.rewards.rewards import RewardRunner


class FailingProvider:
    def compute(self, smiles_list, epoch):
        raise RuntimeError("docking failed")


def test_provider_errors_are_not_converted_to_zero_rewards():
    runner = RewardRunner.__new__(RewardRunner)
    runner.providers = [FailingProvider()]
    runner._decode_selfies = lambda molecules: ["C"] * len(molecules)

    with pytest.raises(RuntimeError, match="docking failed"):
        runner(["[C]"], epoch=0)
