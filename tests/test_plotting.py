import pytest
import pandas as pd
from plotting_utils import make_experiment_plots


@pytest.mark.webtest
class TestPlottingOnDummyData:
    dollarstreet_results = {
        "dollarstreet_test_accuracy": [0.9],
        "dollarstreet-africa_test_accuracy": [0.5],
        "dollarstreet-americas_test_accuracy": [0.4],
        "dollarstreet-asia_test_accuracy": [0.3],
        "dollarstreet-europe_test_accuracy": [0.2],
        "dollarstreet-high_test_accuracy": [0.8],
        "dollarstreet-middle_test_accuracy": [0.7],
    }
    generalization_results = {
        "imageneta_test_accuracy": [0.8],
        "imagenetr_test_accuracy": [0.8],
        "objectnet_test_accuracy": [0.8],
        "imagenetv2_test_accuracy": [0.8],
        "imagenet_test_accuracy": [0.8],
    }
    property_results = {
        "imagenet_sparsity_0.1": [0.8],
        "imagenet_sparsity_0.5": [0.7],
        "imagenet_sparsity_0.1": [0.9],
    }

    results = {}
    results.update(dollarstreet_results)
    results.update(generalization_results)
    results.update(property_results)

    results["Model"] = ["resnet101"]

    results_df = [pd.DataFrame(results)]

    @pytest.mark.parametrize("results_df", results_df)
    def test_generated_correct_number_of_plots(self, results_df):
        figures = make_experiment_plots(results_df, log_dir="")
        assert len(figures) == 4
