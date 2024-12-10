import hydra
from omegaconf import DictConfig
from tqdm.autonotebook import tqdm, trange
import warp as wp
import time

from diffsim import generate_traj, generate_mass_diff, Sim

@hydra.main(config_path="./configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Access parameters from the configuration file
    traj_list = generate_traj(cfg.trajectory)
    mode = 'train'
    mass_diff = generate_mass_diff(cfg.training.mass_diff_config)

    experiment = Sim(cfg, traj_list, device=wp.get_preferred_device(), verbose=True, mass_diff=mass_diff, mode=mode)
    experiment.load_state()

    # Start timing before the loop
    total_start_time = time.time()
    for epoch in trange(cfg.training.train_iters):
        experiment.step()
        # experiment.render()
        tqdm.write('[{}]'.format(experiment.msg))
    
    # Calculate the total elapsed time after the loop ends
    total_elapsed_time = time.time() - total_start_time
    print(f'Total time for {cfg.training.train_iters} iterations: {total_elapsed_time:.4f} seconds')

    experiment.save_state()
    if experiment.renderer:
        experiment.renderer.save()
    experiment.plot_loss()
    experiment.save_training()

if __name__ == "__main__":
    main()