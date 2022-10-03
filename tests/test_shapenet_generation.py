from datasets.shapenet_generation.generate_shapenet import Generation
import pytest
from inspect import getargspec


class TestGeneration:
    def test_instantiation(self):
        generation = Generation()
        assert generation.num_views == 90

    def test_overlapping_synsets(self):
        generation = Generation(synset_type="overlapping")
        assert len(generation.synsets) == 15

    def test_overlapping_synsets_instance_counts(self):
        generation = Generation(synset_type="overlapping")
        assert len(generation.build_commands()) == 50 * 15

    def test_all_synsets(self):
        generation = Generation(synset_type="all")
        # expect 52, since three are excluded (duplicates or timeouts)
        assert len(generation.synsets) == 52
