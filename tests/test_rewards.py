from pockliggpt.rewards.rewards import RewardRunner


class FailingProvider:
    def compute(self, smiles_list, epoch):
        raise RuntimeError("docking failed")


def test_provider_errors_are_converted_to_zero_rewards():
    runner = RewardRunner.__new__(RewardRunner)
    runner.providers = [FailingProvider()]
    runner._decode_selfies = lambda molecules: ["C"] * len(molecules)

    assert runner(["[C]", "[O]"], epoch=0) == [0.0, 0.0]
