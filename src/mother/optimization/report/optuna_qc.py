import logging
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import optuna
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

module_logger: logging.Logger = logging.getLogger(__name__)


def plot_optuna_qc(study: optuna.Study, export_dir: Path, prefix_split: str = "__") -> None:
    """
    Generate and save various Optuna visualizations to the specified directory.

    This method creates and saves the following plots:
    1. Optimization history
    2. Parameter importances
    3. Slice plot
    4. Optimization timeline
    5. Contour plot

    Args:
        export_dir (Path): The directory where the plots will be saved.

    Raises:
        ValueError: If there is an issue plotting parameter importances.
    """
    module_logger.info("Plotting optuna results")
    matplotlib.use("Agg")
    module_logger.debug("Plot optimization history")
    axes: Axes = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(export_dir.joinpath("optuna_history.png"))
    plt.close()
    try:
        module_logger.debug("Plot parameter importances")
        axes = optuna.visualization.matplotlib.plot_param_importances(
            study,
            evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator(),
        )
        # modify y-axis labels
        axes.set_yticklabels([label.get_text().split(prefix_split)[-1] for label in axes.get_yticklabels()])
        plt.tight_layout()
        plt.savefig(export_dir.joinpath("optuna_importance.png"))
        plt.close()
    except ValueError as valError:
        module_logger.error(valError)
        module_logger.warning("Could not plot optuna feature importance values")
    module_logger.debug("Plot slice")
    axis = optuna.visualization.matplotlib.plot_slice(study)
    if isinstance(axis, Iterable):
        for ax in axis:
            ax.set_xlabel(ax.get_xlabel().split(prefix_split)[-1])
    else:
        axis.set_xlabel(axis.get_xlabel().split(prefix_split)[-1])
    # plt.tight_layout()
    plt.savefig(export_dir.joinpath("optuna_slice.png"))
    plt.close()
    module_logger.debug("Plot optimization timeline")
    axes = optuna.visualization.matplotlib.plot_timeline(study)
    axes.set_xlabel(axes.get_xlabel().split(prefix_split)[-1])
    plt.tight_layout()
    plt.savefig(export_dir.joinpath("optuna_timeline.png"))
    plt.close()
    module_logger.debug("Plot contour")
    axis = optuna.visualization.matplotlib.plot_contour(study)
    if isinstance(axis, Iterable):
        for ax_ in axis:
            for ax in ax_:
                ax.set_xlabel(ax.get_xlabel().split(prefix_split)[-1])
                ax.set_ylabel(ax.get_ylabel().split(prefix_split)[-1])
    else:
        axis.set_xlabel(axis.get_xlabel().split(prefix_split)[-1])
        axis.set_ylabel(axis.get_ylabel().split(prefix_split)[-1])
    plt.tight_layout()
    plt.savefig(export_dir.joinpath("optuna_contour.png"))
    plt.close()
