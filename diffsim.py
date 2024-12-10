import torch
import os
import math
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
wp.init()

@wp.kernel
def damp_particle_velocity(
    particle_qd: wp.vec3f,
):
    tid = wp.tid()
    particle_qd[tid] = 0.98 * particle_qd[tid]


class ForwardKinematics(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        body_mass,
        model,
        states,
        sim_dt,
        sim_steps,
        controls,
        update_mass_matrix_every,
        is_colliding,
        particle_damping=False,
    ):

        ctx.tape = wp.Tape()
        ctx.model = model
        # NOTE: update mass (torch -> warp)
        ctx.model.body_mass = wp.from_torch(body_mass)
        ctx.states = states
        ctx.sim_dt = sim_dt
        ctx.sim_steps = sim_steps
        ctx.controls = controls

        with ctx.tape:
            ctx.integrator = wp.sim.FeatherstoneIntegrator(
                ctx.model, update_mass_matrix_every=update_mass_matrix_every
            )

            for i in range(ctx.sim_steps):
                ctx.states[i].clear_forces()
                if is_colliding:
                    wp.sim.collide(ctx.model, ctx.states[i])
                ctx.integrator.simulate(
                    ctx.model,
                    ctx.states[i],
                    ctx.states[i + 1],
                    ctx.sim_dt,
                    ctx.controls[i],
                )

                if particle_damping:
                    wp.launch(
                        kernel=damp_particle_velocity,
                        dim=len(ctx.states[i + 1].particle_qd),
                        inputs=[ctx.states[i + 1].particle_qd],
                        device=ctx.model.device,
                    )

        # NOTE: collect computed joint positions
        joint_q_list = []
        for i in range(ctx.sim_steps):
            joint_q_list.append(wp.to_torch(ctx.states[i].joint_q))
        return tuple(joint_q_list)

    @staticmethod
    def backward(ctx, *adj_joint_q_list):
        for i in range(ctx.sim_steps):
            ctx.states[i].joint_q.grad = wp.from_torch(adj_joint_q_list[i])

        ctx.tape.backward()

        # return adjoint w.r.t. inputs
        return (
            wp.to_torch(ctx.tape.gradients[ctx.model.body_mass]),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class Sim:
    def __init__(
        self, cfg, traj_list, device=None, verbose=False, mass_diff=None, mode=None
    ):
        self.cfg = cfg
        self.traj_name = cfg.trajectory
        self.train_rate = cfg.training.train_rate
        self.save_dir = pathlib.Path(__file__).parent / "experiments" / "log" / self.traj_name / cfg.urdf / mode
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_file_path = self.save_dir / "sim_data.pt"
        if mode == "train" or mode == "test":
            load_dir = pathlib.Path(__file__).parent / "experiments" / "log" / self.traj_name / cfg.urdf / "dataset"
            self.load_file_path = load_dir / "sim_data.pt"
            if mode == "train":
                self.save_stats_path = self.save_dir / "training_stats.pt"
                stage = self.save_dir / "train.usd"
            elif mode == "test":
                self.save_stats_path = self.save_dir / "testing_stats.pt"
                stage = self.save_dir / f"test_ckpt_idx_{cfg.ckpt_idx:04d}.usd"
        else:
            self.load_file_path = self.save_file_path
            stage = self.save_dir / "eval.usd"
        stage = stage.as_posix()
        if cfg.training.load_file_path_overwrite:
            self.load_file_path = cfg.training.load_file_path_overwrite

        self.mode = mode
        self.verbose = verbose
        self.losses = []
        self.masses = []

        articulation_builder = wp.sim.ModelBuilder(gravity=cfg.sim.gravity)

        if hasattr(cfg, "urdf_overwrite"):
            cfg.urdf = cfg.urdf_overwrite

        urdf_path = pathlib.Path(__file__).parent / "assets" / f"{cfg.urdf}.urdf"

        # Import robots unless cfg.sim says not to
        if not hasattr(cfg.sim, "import_robot") or cfg.sim.import_robot:
            wp.sim.parse_urdf(
                urdf_path,
                articulation_builder,
                xform=wp.transform(
                    (0.0, 0.0, 0.0),
                    wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5),
                ),
                floating=False,
                density=cfg.sim.density,
                armature=cfg.sim.armature,
                stiffness=0.0,
                damping=0.0,
                limit_ke=1.0e4,
                limit_kd=1.0e1,
                enable_self_collisions=False,
                parse_visuals_as_colliders=cfg.sim.parse_visuals_as_colliders,
                collapse_fixed_joints=cfg.sim.collapse_fixed_joints,
                ignore_inertial_definitions=cfg.sim.ignore_inertial_definitions,
            )

        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        self.frame_dt = cfg.sim.frame_dt

        episode_duration = cfg.sim.episode_duration  # seconds
        self.episode_frames = int(episode_duration / self.frame_dt)

        self.sim_substeps = cfg.sim.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = traj_list["num_env"]
        num_per_dim = int(math.sqrt(self.num_envs))

        self.control_func = traj_list["control"]

        articulation_builder = modify_builder_with_object(
            articulation_builder, cfg.sim.modify_object_type, cfg
        )

        for id in range(self.num_envs):
            i = int(id / num_per_dim)
            j = id % num_per_dim
            articulation_builder.joint_q = traj_list["q"][id]
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(
                    np.array(((i) * 2.0, (j) * 2.0, 0.0)), wp.quat_identity()
                ),
            )

        # finalize model
        # use `requires_grad=True` to create a model for differentiable simulation
        if cfg.sim.initialization_filename:
            builder = modify_builder_with_joint_data(
                cfg.sim.initialization_filename, builder
            )

        self.model = builder.finalize(device, requires_grad=cfg.sim.requires_grad)
        self.model.ground = False

        self.torch_device = wp.device_to_torch(self.model.device)
        self.renderer = wp.sim.render.SimRenderer(
            path=stage, model=self.model, scaling=15.0
        )

        self.render_time = 0.0
        self.joint_q_list = None

        # optimization variable
        self.body_mass = wp.to_torch(self.model.body_mass, requires_grad=False).clone()
        self.body_mass_single = self.body_mass[0 : len(articulation_builder.body_mass)]

        if cfg.sim.mass_diff:
            mass_diff = cfg.sim.mass_diff

        # noise
        if mass_diff is not None:
            mass_diff = torch.tensor(mass_diff, device=self.torch_device)
            self.one_indices = [
                index for index, value in enumerate(mass_diff) if value == 1
            ]
            self.non_one_indices = [
                index for index, value in enumerate(mass_diff) if value != 1
            ]
            print("ground truth: ", self.body_mass_single[self.non_one_indices])
            self.body_mass_single *= mass_diff
            print("after noise : ", self.body_mass_single[self.non_one_indices])
        else:
            self.one_indices = []
            self.non_one_indices = [
                index for index, _ in enumerate(self.body_mass_single)
            ]

        # override variable with values read from ckpt
        if cfg.ckpt:
            ckpt_path = pathlib.Path(__file__).parent / cfg.ckpt
            ckpt = torch.load(ckpt_path, weights_only=False)
            masses = ckpt["masses"]
            self.body_mass_single = (
                masses[cfg.ckpt_idx].detach().clone()
            )  # second to the last epoch
            print("masses: ", self.body_mass_single)

        self.body_mass_single.requires_grad_()

        self.optimizer = torch.optim.Adam([self.body_mass_single], lr=self.train_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer, T_max=cfg.training.train_iters
        )
        self.criterion = torch.nn.MSELoss()

        self.compare_indices_predictions = slice(
            cfg.training.compare_indices_predictions[0],
            cfg.training.compare_indices_predictions[1],
            cfg.training.compare_indices_predictions[2],
        )

        self.compare_indices_targets = slice(
            cfg.training.compare_indices_targets[0],
            cfg.training.compare_indices_targets[1],
            cfg.training.compare_indices_targets[2],
        )

        self.sim_steps = self.episode_frames * self.sim_substeps
        self.update_mass_matrix_every = (
            self.sim_steps
            if cfg.sim.update_mass_matrix_every == -1
            else cfg.sim.update_mass_matrix_every
        )
        self.is_colliding = cfg.sim.is_colliding
        self.particle_damping = (
            True
            if hasattr(cfg.sim, "particle_damping") and cfg.sim.particle_damping
            else False
        )

    def forward(self):
        # update all the states
        self.body_mass_all = self.body_mass_single.repeat(self.num_envs)
        with torch.no_grad():
            self.model.body_inv_mass = wp.from_torch(1.0 / self.body_mass_all)

        # allocate sim states for trajectory
        self.states = []
        for i in range(self.sim_steps + 1):
            self.states.append(self.model.state())

        self.controls = []
        for i in range(self.sim_steps):
            self.controls.append(self.model.control())
            self.controls[i].joint_act = self.control_func(i)

        self.joint_q_list = ForwardKinematics.apply(
            self.body_mass_all,
            self.model,
            self.states,
            self.sim_dt,
            self.sim_steps,
            self.controls,
            self.update_mass_matrix_every,
            self.is_colliding,
            self.particle_damping,
        )

    def compute_loss(self):
        predictions = torch.cat(self.joint_q_list)
        targets = torch.cat(self.load_data["joint_q_list"]).to(predictions.device)
        self.loss = self.criterion(
            predictions[self.compare_indices_predictions],
            targets[self.compare_indices_targets],
        )

    def step(self):
        def closure():
            self.forward()
            self.compute_loss()
            self.loss.backward()

            self.body_mass_single.grad[self.one_indices] = 0  # fixed object

            self.msg = "loss: {loss}, loss grad: {loss_grad}, masses: {masses}".format(
                loss=self.loss.item(),
                loss_grad=self.body_mass_single.grad[self.non_one_indices],
                masses=self.body_mass_single[self.non_one_indices],
            )

            # Append the info to the list
            self.losses.append(self.loss.item())
            self.masses.append(self.body_mass_single.clone())

            return self.loss.item()  # Return loss value

        # Perform optimization step
        loss = self.optimizer.step(closure)
        self.optimizer.zero_grad()
        self.lr_scheduler.step()

    def render(self):
        if self.renderer is None:
            return
        frame_count = 0
        print("render begin.")
        for i in range(0, self.sim_steps):
            if i % (self.sim_substeps * self.cfg.render.every_n_frame) == 0:
                self.renderer.begin_frame(self.render_time)

                self.renderer.render(self.states[i])

                self.renderer.end_frame()
                self.render_time += self.frame_dt

                cfg = dict(zip(self.model.joint_name, self.states[i].joint_q.numpy()))
                print(self.states[i].joint_q)
                if self.cfg.output_obj:
                    self.output_obj.output(cfg, frame_count)
                frame_count += 1

        frame_idx = 0
        frame_time = 0.0
        for i in range(0, self.sim_steps):
            if i % (self.sim_substeps) == 0:
                frame_idx += 1
                frame_time += self.frame_dt
                name = f"sim_{frame_idx:04d}.pt"
                save_data = {
                    "time": frame_time,
                    "joint_q": wp.to_torch(self.states[i].joint_q),
                }
                save_dir = self.save_dir / "sim_data"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(save_data, save_dir / name)
        print("render finish.")

    def save_state(self, save=True):
        with torch.no_grad():
            self.save_data = {
                "joint_q_list": self.joint_q_list,
            }
        if save:
            torch.save(self.save_data, self.save_file_path)

    def save_training(self):
        # Find the index of the smallest loss
        min_loss_index = self.losses.index(min(self.losses))
        # Get the smallest loss and its corresponding mass
        smallest_loss = self.losses[min_loss_index]
        corresponding_mass = self.masses[min_loss_index]

        print(
            "min_loss_index, smallest_loss, corresponding_mass: ",
            min_loss_index,
            smallest_loss,
            corresponding_mass,
        )

        with torch.no_grad():
            self.save_stats = {
                "losses": self.losses,
                "masses": self.masses,
            }
        torch.save(self.save_stats, self.save_stats_path)
        print("saved training: ", self.save_stats_path)

    def save_testing(self):
        with torch.no_grad():
            self.save_stats = {
                "loss": self.loss,
            }
        print("[loss: {loss}]".format(loss=self.loss.item()))
        print("saved testing: ", self.save_stats_path)

    def load_state(self):
        file = pathlib.Path(self.load_file_path)
        if file.is_absolute():
            file = file.as_posix()
        else:
            file = pathlib.Path(__file__).parent / file
        self.load_data = torch.load(file, weights_only=False)

    def plot_loss(self):
        plt.figure(figsize=(6, 4))
        plt.plot(
            self.losses,
            linestyle="--",
            label="Difference between simulation and observation",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            self.save_dir / f"loss_vs_iter_{self.traj_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

def generate_traj(traj_config):
    if traj_config == 'robotis_2_hard_ball':
        traj_list = {}
        num_env = 1
        q_list = [[0.0]]
        act_list_flat = [-0.49]
        def control_func(step):
            act = wp.array(act_list_flat, dtype=float, requires_grad=True)
            return act
        traj_list['num_env'] = num_env
        traj_list['q'] = q_list
        traj_list['control'] = control_func

        return traj_list
    else:
        raise ValueError(f"Invalid traj_config: {traj_config}")

def generate_mass_diff(mass_diff_config):
    if mass_diff_config == 'cup_diff_none':
        mass_diff = [1,1,1,1,1,1,1,1,1.0001]
        return mass_diff
    else:
        raise ValueError(f'Invalid mass_diff_config: {mass_diff_config}.')

def modify_builder_with_joint_data(file, builder):
    file = pathlib.Path(file)
    if file.is_absolute():
        file = file.as_posix()
    else:
        file = pathlib.Path(__file__).parent / file
    # Load data from the file
    data = torch.load(file, weights_only=False)
    joint_q = data["joint_q"]

    # Apply offset to joint_q[2]
    offset = -np.radians(90.0)
    joint_q[2] += offset

    # Limit joint_q to the first four joints
    joint_q = joint_q[:4]

    # Modify builder's joint_X_p based on joint names
    for idx, transform in enumerate(builder.joint_X_p):
        if builder.joint_name[idx] == "joint1":
            joint_axis = wp.vec3(0.0, 0.0, 1.0)
            angle = joint_q[0]
        elif builder.joint_name[idx] == "joint2":
            joint_axis = wp.vec3(0.0, 1.0, 0.0)
            angle = joint_q[1]
        elif builder.joint_name[idx] == "joint3":
            joint_axis = wp.vec3(0.0, 1.0, 0.0)
            angle = joint_q[2]
        elif builder.joint_name[idx] == "joint4":
            joint_axis = wp.vec3(0.0, 1.0, 0.0)
            angle = joint_q[3]
        else:
            joint_axis = None
            angle = None

        # If joint_axis is valid, update the transform in the builder
        if joint_axis is not None:
            rot = wp.quat_from_axis_angle(joint_axis, float(angle))
            builder.joint_X_p[idx] = wp.transform(
                wp.transform_get_translation(transform), rot
            )

    return builder

def modify_builder_with_object(builder, modify_object_type, cfg):
    if modify_object_type == "hard_ball":
        b = builder.add_body()

        builder.add_shape_sphere(
            body=b, radius=0.0225, density=10, has_shape_collision=False
        )

        builder.add_joint_fixed(
            parent=b - 1,
            child=b,
        )

        return builder
    elif modify_object_type == None:
        return builder
    else:
        raise ValueError(f"Invalid modify_object_type: {modify_object_type}.")
