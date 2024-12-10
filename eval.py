import hydra
from omegaconf import DictConfig
import warp as wp
import time

from diffsim import Sim, generate_traj

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Access parameters from the configuration file
    traj_list = generate_traj(cfg.trajectory)
    mode = "dataset" if cfg.ckpt is None else "test"

    experiment = Sim(
        cfg,
        traj_list,
        device=wp.get_preferred_device(),
        verbose=True,
        mass_diff=None,
        mode=mode,
    )

    total_start_time = time.time()
    experiment.forward()
    total_elapsed_time = time.time() - total_start_time
    print(f"Total time: {total_elapsed_time:.4f} seconds")

    if mode == "test":
        experiment.load_state()
        experiment.compute_loss()
        experiment.save_testing()
    experiment.render()
    experiment.save_state()

    if experiment.renderer:
        experiment.renderer.save()

if __name__ == "__main__":
    main()
