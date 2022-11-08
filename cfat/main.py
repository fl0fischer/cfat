import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy import stats
import os, glob
import logging
from functools import reduce
import mujoco_py

# Local imports
from cfat.utils import muscle_activation_model_secondorder, muscle_activation_model_secondorder_inverse, \
    store_trajectories_table


def _create_mujoco_sim(model_filename, use_mujoco_py=False):
    if use_mujoco_py:
        model = mujoco_py.load_model_from_path(model_filename)
        sim = mujoco_py.MjSim(model)
    else:
        raise NotImplementedError

    return sim


def CFAT_algorithm(table_filename,
                   model_filename,
                   physical_joints=None,
                   param_t_activation=0.04,
                   param_t_excitation=0.03,
                   submovement_times=None,  # optional: compute CFAT only for certain part of table_filename
                   num_target_switches=None,
                   ensure_constraints=False,
                   reset_pos_and_vel=True,
                   usemuscles=False,
                   useexcitationcontrol=True,
                   optimize_excitations=False,
                   usecontrollimits=False,
                   only_initial_values=False,
                   use_qacc=False,
                   timestep=0.002,
                   allow_smaller_data_timesteps=True,
                   results_dir=None,
                   use_mujoco_py=True,
                   store_results_to_csv=True,
                   store_mj_inverse=True):
    output_table_filename = os.path.join(results_dir,
                                         os.path.split(os.path.splitext(table_filename)[0])[1].split('_PhaseSpace_')[
                                             -1] + '_CFAT.csv')

    if table_filename.endswith('.csv'):
        trajectories_table = pd.read_csv(table_filename, index_col="time")
    else:
        trajectories_table = pd.read_csv(table_filename, skiprows=10, delimiter="\t", index_col="time")

    # Create MuJoCo environment
    sim = _create_mujoco_sim(model_filename, use_mujoco_py=use_mujoco_py)
    if physical_joints is None:
        physical_joint_ids = list(range(sim.model.njnt))
        virtual_joint_ids = []
        virtual_joints = []
        if use_mujoco_py:
            physical_joints = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
        else:
            physical_joints = [sim.model.joint(i).name for i in range(sim.model.njnt)]
    else:
        if use_mujoco_py:
            physical_joint_ids = [sim.model.joint_name2id(name) for name in physical_joints]
            virtual_joint_ids = [i for i in range(sim.model.njnt) if (i not in physical_joint_ids)]
            virtual_joints = [sim.model.joint_id2name(i) for i in virtual_joint_ids]
        else:
            physical_joint_ids = [sim.model.joint(name).id for name in physical_joints]
            virtual_joint_ids = [i for i in range(sim.model.njnt) if (i not in physical_joint_ids)]
            virtual_joints = [sim.model.joint(i).name for i in virtual_joint_ids]

    # Remove gears and control limits from used model
    sim.model.actuator_gear[:] = np.array([[1, 0, 0, 0, 0, 0]] * sim.model.nu)
    sim.model.actuator_ctrllimited[:] = False
    sim.model.actuator_ctrlrange[:] = np.array([[-100, 100]] * sim.model.nu)
    sim.model.actuator_biastype[:] = False

    if submovement_times is not None:
        assert len(submovement_times) == 5  # 2
        trajectories_table[["Init_ID", "Target_ID", "Movement_ID"]] = submovement_times[2:].astype(int)
        # submovement_times = trajectories_table.iloc[submovement_times].index  #if submovement_times incorporates indices instead of time values
    else:
        trajectories_table = trajectories_table.loc[trajectories_table.index >= trajectories_table.index[
            0] - trajectories_table.index.to_series().diff().mean(),
                             :].rename(lambda i: i + '_pos', axis=1) * np.pi / 180
    """
    INFO - USED SAMPLING INTERVALS:
    - system dynamics/internal MuJoCo sample interval length: sim.model.opt.timestep
    - (interpolated) data sampling interval length: timestep_data (= frame_skip_data * sim.model.opt.timestep)
    - control sample interval length (length of interval with constant control): timestep (= simulation_frame_skip * sim.model.opt.timestep)

    --> NECESSARY CONDITIONS: sim.model.opt.timestep <= timestep_data <= timestep, and all three ratios need to be Integer
    --> given sim.model.opt.timestep (from MuJoCo model) and timestep (as argument), timestep_data is computed as value
        that is closest to original data sampling interval length and satifies all necessary conditions

    WARNING: reasonable qpos and qvel fits (with optimize_excitations=False) were obtained with 
    sim.model.opt.timestep=0.002, timestep_data=0.004, and timestep=0.04. 
    In particular, for different data sampling rate, i.e., different timestep_data,
    used meta-parameters such as the loss function weights and parameters of Savitzky-Golay filter 
    applied to ground-truth data might need to be tuned manually again...
    """

    def factors(n):
        return np.array(reduce(list.__add__,
                               ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    # SIMULATE USING MULTIPLE MUJOCO STEPS PER INPUT ROW:
    assert (timestep / sim.model.opt.timestep).is_integer()
    simulation_frame_skip = int(timestep / sim.model.opt.timestep)
    possible_factors = factors(simulation_frame_skip)
    if not allow_smaller_data_timesteps:  # only allow "timestep_data" greater or equal (original) "trajectories_table.index.to_series().diff().mean()"
        possible_factors = possible_factors[possible_factors >= int(
            trajectories_table.index.to_series().diff().mean() // sim.model.opt.timestep)]
    frame_skip_data = possible_factors[np.argmin(np.abs(
        possible_factors * sim.model.opt.timestep - trajectories_table.index.to_series().diff().mean()))]
    timestep_data = frame_skip_data * sim.model.opt.timestep
    timestep_data = np.round(timestep_data, 8)  # get rid of rounding errors...
    assert (timestep / timestep_data).is_integer()
    ### INTERPOLATE DATA FILE:
    trajectories_table = trajectories_table.reindex(
        trajectories_table.index.union(
            np.arange(trajectories_table.index[0], trajectories_table.index[-1], timestep_data))).interpolate()
    trajectories_table = trajectories_table.loc[
                         np.arange(trajectories_table.index[0], trajectories_table.index[-1], timestep_data), :]
    if submovement_times is not None:  # convert added columns with submovement information back to integer
        trajectories_table[[cn for cn in trajectories_table.columns if cn.endswith('_ID')]] = trajectories_table[
            [cn for cn in trajectories_table.columns if cn.endswith('_ID')]].astype(int)

    # Remove last frames from table which cannot be grouped into one complete timestep:
    if int((trajectories_table.shape[0] - 1) % (timestep / timestep_data)):
        trajectories_table = trajectories_table.iloc[
                             :-int((trajectories_table.shape[0] - 1) % (timestep / timestep_data))]
    print((sim.model.opt.timestep, frame_skip_data, timestep_data, simulation_frame_skip, timestep))

    trajectories_table = trajectories_table.rename(lambda i: i - trajectories_table.index[0], axis=0)
    trajectories_table.index.names = ["time"]
    # trajectories_table = trajectories_table.loc[(trajectories_table.index <= nseconds), :]

    missing_columns = ['shoulder_elv_pos',
                       'shoulder_rot_pos',
                       'elbow_flexion_pos',
                       'pro_sup_pos',
                       'deviation_pos',
                       'flexion_pos',
                       'elv_angle_vel',
                       'shoulder_elv_vel',
                       'shoulder_rot_vel',
                       'elbow_flexion_vel',
                       'pro_sup_vel',
                       'deviation_vel',
                       'flexion_vel',
                       'thorax_xpos_x',
                       'thorax_xpos_y',
                       'thorax_xpos_z',
                       'humerus_xpos_x',
                       'humerus_xpos_y',
                       'humerus_xpos_z',
                       'ulna_xpos_x',
                       'ulna_xpos_y',
                       'ulna_xpos_z',
                       'radius_xpos_x',
                       'radius_xpos_y',
                       'radius_xpos_z',
                       'proximal_row_xpos_x',
                       'proximal_row_xpos_y',
                       'proximal_row_xpos_z',
                       'hand_xpos_x',
                       'hand_xpos_y',
                       'hand_xpos_z',
                       'end-effector_xpos_x',
                       'end-effector_xpos_y',
                       'end-effector_xpos_z',
                       'target_xpos_x',
                       'target_xpos_y',
                       'target_xpos_z',
                       'thorax_xvelp_x',
                       'thorax_xvelp_y',
                       'thorax_xvelp_z',
                       'humerus_xvelp_x',
                       'humerus_xvelp_y',
                       'humerus_xvelp_z',
                       'ulna_xvelp_x',
                       'ulna_xvelp_y',
                       'ulna_xvelp_z',
                       'radius_xvelp_x',
                       'radius_xvelp_y',
                       'radius_xvelp_z',
                       'proximal_row_xvelp_x',
                       'proximal_row_xvelp_y',
                       'proximal_row_xvelp_z',
                       'hand_xvelp_x',
                       'hand_xvelp_y',
                       'hand_xvelp_z',
                       'end-effector_xvelp_x',
                       'end-effector_xvelp_y',
                       'end-effector_xvelp_z',
                       'target_xvelp_x',
                       'target_xvelp_y',
                       'target_xvelp_z',
                       'accsensor_end-effector_x',
                       'accsensor_end-effector_y',
                       'accsensor_end-effector_z',
                       'difference_vec_x',
                       'difference_vec_y',
                       'difference_vec_z',
                       'centroid_vel_projection',
                       'target_width',
                       'A_elv_angle',
                       'A_shoulder_elv',
                       'A_shoulder_rot',
                       'A_elbow_flexion',
                       'A_pro_sup',
                       'A_deviation',
                       'A_flexion',
                       'thorax_tx_frc',
                       'thorax_ty_frc',
                       'thorax_tz_frc',
                       'thorax_rx_frc',
                       'thorax_ry_frc',
                       'thorax_rz_frc',
                       'sternoclavicular_r2_frc',
                       'sternoclavicular_r3_frc',
                       'unrotscap_r3_frc',
                       'unrotscap_r2_frc',
                       'acromioclavicular_r2_frc',
                       'acromioclavicular_r3_frc',
                       'acromioclavicular_r1_frc',
                       'unrothum_r1_frc',
                       'unrothum_r3_frc',
                       'unrothum_r2_frc',
                       'elv_angle_frc',
                       'shoulder_elv_frc',
                       'shoulder1_r2_frc',
                       'shoulder_rot_frc',
                       'elbow_flexion_frc',
                       'pro_sup_frc',
                       'deviation_frc',
                       'flexion_frc',
                       'wrist_hand_r1_frc',
                       'wrist_hand_r3_frc',
                       'reward',
                       'step_type',
                       'target_switch',
                       'discount']

    missing_columns_true = [cn for cn in missing_columns if (cn not in trajectories_table.columns)]
    trajectories_table = pd.concat((trajectories_table, pd.DataFrame(columns=missing_columns_true)), axis=1)
    trajectories_table.index.names = ["time"]
    trajectories_table.loc[:, [cn for cn in missing_columns_true if not cn.startswith("A_")]] = 0.0

    trajectories_table = trajectories_table.reset_index()

    # TODO: delete
    # trajectories_table.loc[:, "thorax_xpos_x":"thorax_xpos_z"] -= sim.model.body_pos[sim.model.body_name2id("thorax")]
    # sim.model.body_pos[sim.model.body_name2id("thorax")] = np.array([0, 0, 0])

    # SIMULATE USING ONE MUJOCO STEP PER INPUT ROW:
    # simulation_frame_skip = 1
    # print(f'INFO: Settings MuJoCo timestep to {trajectories_table.time.diff().mean()}s.')
    # sim.model.opt.timestep = trajectories_table.time.diff().mean() / 1

    # SIMULATE USING MULTIPLE MUJOCO STEPS PER INPUT ROW:
    #### See above... ####

    goal_coordinates = trajectories_table.loc[:, "target_xpos_x":"target_xpos_z"].iloc[0].values
    assert trajectories_table.target_width.unique().shape == (1,)
    goal_width = trajectories_table.target_width.iloc[0]

    target_switch_indices = trajectories_table.loc[trajectories_table.target_switch == 1].index

    # goal = (goal_coordinates, goal_width)
    # simulator.skip_target_choice = True  # goal is set manually

    if submovement_times is None:  # otherwise, input table is assumed to incorporate filtered data
        for column in [column_name for column_name in trajectories_table.columns if column_name.endswith('_pos')]:
            # pass
            trajectories_table[column] = trajectories_table[[column]].apply(
                lambda x: savgol_filter(x, 15, 3, deriv=0, delta=timestep_data))
        for column in [column_name for column_name in trajectories_table.columns if column_name.endswith('_pos')]:
            trajectories_table[column[:-4] + "_vel"] = trajectories_table[[column[:-4] + "_pos"]].apply(
                lambda x: savgol_filter(x, 15, 3, deriv=1, delta=timestep_data))
        for column in [column_name for column_name in trajectories_table.columns if column_name.endswith('_pos')]:
            trajectories_table[column[:-4] + "_acc"] = trajectories_table[[column[:-4] + "_pos"]].apply(
                lambda x: savgol_filter(x, 15, 3, deriv=2, delta=timestep_data))

    # Only use part of trajectory file (while smoothening the whole file above)
    if submovement_times is not None:
        trajectories_table_test = trajectories_table.loc[(trajectories_table.time >= submovement_times[0]) & (
                trajectories_table.time <= submovement_times[1]), :].reset_index(drop=True)
        if trajectories_table_test.shape[0] < 2 * int(
                timestep / timestep_data) + 1:  # use next time step, even though it is not in the desired time range
            print(
                f"WARNING: After interpolation, too few time steps are within the desired movement time range. Will use next additional time steps.")
            trajectories_table = trajectories_table.loc[(trajectories_table.time >= submovement_times[0]), :].iloc[
                                 :2 * int(timestep / timestep_data) + 1, :].reset_index(drop=True)
        else:
            trajectories_table = trajectories_table_test

    # Posture initialization:
    qpos = np.zeros((sim.model.nq,))  # sim.data.qpos.copy()
    qvel = np.zeros((sim.model.nq,))  # sim.data.qvel.copy()
    qacc = np.zeros((sim.model.nq,))  # sim.data.qacc.copy()
    for column_name in [i.split('_pos')[0] for i in trajectories_table.columns if
                        i.endswith(
                            '_pos')]:  # & ~i.startswith('thorax') & ((not usemuscles) or ~i.startswith('thorax'))]:
        qpos[sim.model.joint_name2id(column_name)] = trajectories_table.loc[:, column_name + '_pos'].iloc[
            0]
    for column_name in [i.split('_vel')[0] for i in trajectories_table.columns if
                        i.endswith(
                            '_vel')]:  # & ~i.startswith('thorax') & ((not usemuscles) or ~i.startswith('thorax'))]:
        qvel[sim.model.joint_name2id(column_name)] = trajectories_table.loc[:, column_name + '_vel'].iloc[
            0]
    for column_name in [i.split('_acc')[0] for i in trajectories_table.columns if
                        i.endswith(
                            '_acc')]:  # & ~i.startswith('thorax') & ((not usemuscles) or ~i.startswith('thorax'))]:
        qacc[sim.model.joint_name2id(column_name)] = trajectories_table.loc[:, column_name + '_acc'].iloc[
            0]

    # Set initial thorax translation:
    remaining_thorax_translation = trajectories_table.loc[:, "thorax_xpos_x":"thorax_xpos_z"].iloc[0] - \
                                   sim.model.body_pos[sim.model.body_name2id("thorax")]
    qpos[sim.model.joint_name2id("thorax_tx")] = remaining_thorax_translation[0]
    qpos[sim.model.joint_name2id("thorax_ty")] = remaining_thorax_translation[1]
    qpos[sim.model.joint_name2id("thorax_tz")] = remaining_thorax_translation[2]
    qvel[sim.model.joint_name2id("thorax_tx")] = 0
    qvel[sim.model.joint_name2id("thorax_ty")] = 0
    qvel[sim.model.joint_name2id("thorax_tz")] = 0

    # adjust thorax constraints to keep initial thorax posture:
    for column_name in ["thorax_tx", "thorax_ty", "thorax_tz", "thorax_rx", "thorax_ry", "thorax_rz"]:
        sim.model.eq_data[
            (sim.model.eq_obj1id[:] == sim.model.joint_name2id(column_name)) & (
                    sim.model.eq_type[:] == 2), 0] = qpos[
            sim.model.joint_name2id(column_name)]
    if ensure_constraints:
        # adjust virtual joints according to active constraints:
        for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
                sim.model.eq_obj1id[
                    (sim.model.eq_type == 2) & (sim.model.eq_active == 1)],
                sim.model.eq_obj2id[
                    (sim.model.eq_type == 2) & (sim.model.eq_active == 1)],
                sim.model.eq_data[(sim.model.eq_type == 2) &
                                  (sim.model.eq_active == 1), 4::-1]):
            qpos[virtual_joint_id] = np.polyval(poly_coefs, qpos[physical_joint_id])

    param_init_qpos = qpos
    param_init_qvel = qvel
    # simulator.reset()  # env_settings_reset=env_settings_reset)
    if use_mujoco_py:
        sim.reset()
        x0 = sim.get_state()
        x0.qpos[:] = param_init_qpos
        x0.qvel[:] = param_init_qvel
        sim.set_state(x0)
    else:
        raise NotImplementedError

    if useexcitationcontrol and optimize_excitations:
        activations = np.zeros(sim.model.nu, )
        activations_first_derivative = np.zeros(sim.model.nu, )

    output_table = trajectories_table.copy()  # .set_index('time')
    output_table = output_table.rename(lambda x: x + '_orig' if x[-4:] in ['_pos', '_vel', '_acc'] else x, axis=1)

    for column_name in physical_joints + virtual_joints:
        output_table.loc[output_table.index[0], column_name + '_pos'] = sim.data.qpos[
            sim.model.joint_name2id(column_name)]
        output_table.loc[output_table.index[0], column_name + '_vel'] = sim.data.qvel[
            sim.model.joint_name2id(column_name)]
        output_table.loc[output_table.index[0], column_name + '_acc'] = sim.data.qacc[
            sim.model.joint_name2id(column_name)]

    last_index = trajectories_table.iloc[:-1:int(timestep / timestep_data), :].index[-1]

    for (index, row), (_, nextrow) in zip(trajectories_table.iloc[:-1:int(timestep / timestep_data), :].iterrows(),
                                          trajectories_table.iloc[1:, :].groupby(
                                              np.arange(len(trajectories_table) - 1) // int(
                                                  timestep / timestep_data))):
        percentage_achieved = float(index * 100 / last_index) if last_index != 0 else 100
        print('Optimization - Time Step #{}/{} [{:.2f}%]'.format(index, last_index, percentage_achieved))

        def _visualize_cond_fn(index, num_target_switches):
            if num_target_switches is not None:
                return (index < target_switch_indices[
                    num_target_switches])  # visualize exactly num_target_switches movements
            else:
                return True

        if _visualize_cond_fn(index, num_target_switches):
            last_qpos_data = sim.data.qpos.copy()
            last_qvel_data = sim.data.qvel.copy()
            next_qpos_data = np.zeros(shape=(int(timestep / timestep_data), sim.model.nq))
            next_qvel_data = np.zeros(shape=(int(timestep / timestep_data), sim.model.nq))
            next_qacc_data = np.zeros(shape=(int(timestep / timestep_data), sim.model.nq))
            forward_qpos = sim.data.qpos.copy()
            forward_qvel = sim.data.qvel.copy()
            if useexcitationcontrol and optimize_excitations:
                forward_activations = activations.copy()
                forward_activations_first_derivative = activations_first_derivative.copy()
                # input((forward_activations, forward_activations_first_derivative))
            # input((row[[column_name + '_pos' for column_name in physical_joints]], nextrow[[column_name + '_pos' for column_name in physical_joints]]))
            for column_name in [i.split('_pos')[0] for i in trajectories_table.columns if
                                i.endswith('_pos')]:  # & ~i.startswith('thorax')]:
                last_qpos_data[sim.model.joint_name2id(column_name)] = row[column_name + '_pos']
            for column_name in [i.split('_vel')[0] for i in trajectories_table.columns if
                                i.endswith('_vel')]:  # & ~i.startswith('thorax')]:
                last_qvel_data[sim.model.joint_name2id(column_name)] = row[column_name + '_vel']
            for next_indices in range(nextrow.shape[0]):  # range(int(timestep / timestep_data)):
                for column_name in [i.split('_pos')[0] for i in trajectories_table.columns if
                                    i.endswith('_pos')]:  # & ~i.startswith('thorax')]:
                    next_qpos_data[next_indices][sim.model.joint_name2id(column_name)] = \
                        nextrow[column_name + '_pos'].iloc[next_indices]
                for column_name in [i.split('_vel')[0] for i in trajectories_table.columns if
                                    i.endswith('_vel')]:  # & ~i.startswith('thorax')]:
                    next_qvel_data[next_indices][sim.model.joint_name2id(column_name)] = \
                        nextrow[column_name + '_vel'].iloc[next_indices]
                for column_name in [i.split('_acc')[0] for i in trajectories_table.columns if
                                    i.endswith('_acc')]:  # & ~i.startswith('thorax')]:
                    next_qacc_data[next_indices][sim.model.joint_name2id(column_name)] = \
                        nextrow[column_name + '_acc'].iloc[next_indices]
            if ensure_constraints:
                # adjust thorax constraints to keep initial thorax posture:
                for column_name in ["thorax_tx", "thorax_ty", "thorax_tz", "thorax_rx", "thorax_ry",
                                    "thorax_rz"]:
                    sim.model.eq_data[
                        (sim.model.eq_obj1id[:] == sim.model.joint_name2id(
                            column_name)) & (
                                sim.model.eq_type[:] == 2), 0] = qpos[
                        sim.model.joint_name2id(column_name)]

                # adjust virtual joints according to active constraints:
                for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
                        sim.model.eq_obj1id[
                            (sim.model.eq_type == 2) & (sim.model.eq_active == 1)],
                        sim.model.eq_obj2id[
                            (sim.model.eq_type == 2) & (sim.model.eq_active == 1)],
                        sim.model.eq_data[
                        (sim.model.eq_type == 2) & (sim.model.eq_active == 1), 4::-1]):
                    # print((virtual_joint_id, np.abs(np.polyval(poly_coefs, qpos[physical_joint_id]) - forward_qpos[virtual_joint_id])))
                    if reset_pos_and_vel:
                        last_qpos_data[virtual_joint_id] = np.polyval(poly_coefs, last_qpos_data[physical_joint_id])
                    else:
                        forward_qpos[virtual_joint_id] = np.polyval(poly_coefs, forward_qpos[physical_joint_id])
            if reset_pos_and_vel:
                param_init_qpos = last_qpos_data
                param_init_qvel = last_qvel_data
            else:
                param_init_qpos = forward_qpos
                param_init_qvel = forward_qvel
            if reset_pos_and_vel:
                # simulator.reset()  # env_settings_reset=env_settings_reset)  #WARNING: This sets qpos and qvel from data, which might lead to accumulating errors during long step-wise optimizations...
                if use_mujoco_py:
                    sim.reset()
                    x0 = sim.get_state()
                    x0.qpos[:] = param_init_qpos
                    x0.qvel[:] = param_init_qvel
                    sim.set_state(x0)
                else:
                    raise NotImplementedError
            # Write action sequence consisting of zeros into output_table...
            action = np.zeros(sim.model.nu, )

            def minimize_posvelacc_error(ctrl, pos, vel, acc):
                pos_error = 0
                vel_error = 0
                acc_error = 0

                for frame_skip_index in range(simulation_frame_skip):
                    if not use_qacc:
                        forward_qvel_testing = sim.data.qvel.copy()

                    sim.data.ctrl[:] = ctrl
                    try:
                        sim.step()
                    except mujoco_py.builder.MujocoException:
                        print(
                            f"ERROR: The simulation is unstable.\n\tqpos={sim.data.qpos[:]}\n\tqvel={sim.data.qvel[:]}\n\tctrl={ctrl}")
                        pos_error = np.inf
                        vel_error = np.inf
                        acc_error = np.inf
                        break
                    if (frame_skip_index + 1) % frame_skip_data == 0:
                        pos_error += np.linalg.norm(
                            sim.data.qpos[physical_joint_ids] -
                            pos[frame_skip_index // frame_skip_data][
                                physical_joint_ids])
                        vel_error += np.linalg.norm(
                            sim.data.qvel[physical_joint_ids] -
                            vel[frame_skip_index // frame_skip_data][
                                physical_joint_ids])
                        if use_qacc:
                            # TODO: do we need reserve actuators as in OpenSim (if only 6 out of 7 controls are directly penalized with acc costs, the remaining control seems to take this function...)
                            acc_error += np.linalg.norm(
                                sim.data.qacc[physical_joint_ids] -
                                acc[frame_skip_index // frame_skip_data][
                                    physical_joint_ids])
                        else:
                            acc_error += np.linalg.norm(
                                ((sim.data.qvel[physical_joint_ids] -
                                  forward_qvel_testing[
                                      physical_joint_ids]) / sim.model.opt.timestep) -
                                acc[frame_skip_index // frame_skip_data][
                                    physical_joint_ids])

                pos_error /= int(simulation_frame_skip / frame_skip_data)
                vel_error /= int(simulation_frame_skip / frame_skip_data)
                acc_error /= int(simulation_frame_skip / frame_skip_data)

                # simulator.reset()
                if use_mujoco_py:
                    sim.reset()
                    x0 = sim.get_state()
                    x0.qpos[:] = param_init_qpos
                    x0.qvel[:] = param_init_qvel
                    sim.set_state(x0)
                    sim.forward()
                else:
                    raise NotImplementedError

                if usemuscles:
                    return vel_error
                else:
                    if only_initial_values:
                        return acc_error  # initial activations (e.g., used for MPC) computed by CFAT should mainly match initial acceleration
                    else:
                        return 1000 * pos_error + 50 * vel_error + 0.01 * acc_error

            def minimize_posvelacc_error_secondorder_control(ctrl, pos, vel, acc):
                global activations
                global activations_first_derivative

                pos_error = 0
                vel_error = 0
                acc_error = 0

                for frame_skip_index in range(simulation_frame_skip):
                    activations, activations_first_derivative = muscle_activation_model_secondorder(activations,
                                                                                                    activations_first_derivative,
                                                                                                    ctrl,
                                                                                                    timestep,
                                                                                                    param_t_activation,
                                                                                                    param_t_excitation)

                    sim.data.ctrl[:] = activations
                    input((simulation_frame_skip,
                           simulation_frame_skip))  # delete this if both match (otherwise, there might be a problem with sim.step()...)
                    sim.step()

                    if (frame_skip_index + 1) % frame_skip_data == 0:
                        pos_error += np.linalg.norm(
                            sim.data.qpos[physical_joint_ids] -
                            pos[frame_skip_index // frame_skip_data][
                                physical_joint_ids])
                        vel_error += np.linalg.norm(
                            sim.data.qvel[physical_joint_ids] -
                            vel[frame_skip_index // frame_skip_data][
                                physical_joint_ids])
                        acc_error += np.linalg.norm(
                            sim.data.qacc[physical_joint_ids] -
                            acc[frame_skip_index // frame_skip_data][
                                physical_joint_ids])

                pos_error /= int(simulation_frame_skip / frame_skip_data)
                vel_error /= int(simulation_frame_skip / frame_skip_data)
                acc_error /= int(simulation_frame_skip / frame_skip_data)

                # simulator.reset()
                if use_mujoco_py:
                    sim.reset()
                    x0 = sim.get_state()
                    x0.qpos[:] = param_init_qpos
                    x0.qvel[:] = param_init_qvel
                    sim.set_state(x0)
                else:
                    raise NotImplementedError

                activations = forward_activations
                activations_first_derivative = forward_activations_first_derivative

                return 1000 * pos_error + 50 * vel_error + 0.01 * acc_error  # + activations_penalty

            if useexcitationcontrol and optimize_excitations:
                if usecontrollimits:
                    feasible_controls_sol = minimize(minimize_posvelacc_error_secondorder_control, action,
                                                     args=(next_qpos_data, next_qvel_data, next_qacc_data),
                                                     bounds=sim.model.actuator_ctrlrange, method="SLSQP")
                else:
                    feasible_controls_sol = minimize(minimize_posvelacc_error_secondorder_control, action,
                                                     args=(next_qpos_data, next_qvel_data, next_qacc_data),
                                                     method="SLSQP")
            elif usemuscles:
                # feasible_controls_sol = minimize(minimize_posvelacc_error, action, args=(next_qpos_data, next_qvel_data, next_qacc_data), bounds=sim.model.actuator_ctrlrange, method="SLSQP")
                feasible_controls_sol = minimize(minimize_posvelacc_error, action,
                                                 args=(next_qpos_data, next_qvel_data, next_qacc_data),
                                                 bounds=[(1e-8, 100) for _ in range(sim.model.nu)],
                                                 method="SLSQP")
            else:
                if usecontrollimits:
                    feasible_controls_sol = minimize(minimize_posvelacc_error, action,
                                                     args=(next_qpos_data, next_qvel_data, next_qacc_data),
                                                     bounds=sim.model.actuator_ctrlrange, method="SLSQP")
                else:
                    feasible_controls_sol = minimize(minimize_posvelacc_error, action,
                                                     args=(next_qpos_data, next_qvel_data, next_qacc_data),
                                                     method="SLSQP")

            ctrl = feasible_controls_sol.x

            next_index = [i for i in output_table.index if i > output_table.index[index]][0]
            if usemuscles:
                for actuator_name in [sim.model.actuator_id2name(actuator_id) for actuator_id in
                                      range(sim.model.nu)]:
                    output_table.loc[index, 'A_' + actuator_name] = ctrl[
                        sim.model.actuator_name2id(actuator_name)]
            else:
                for column_name in physical_joints:
                    output_table.loc[index, 'A_' + column_name] = ctrl[
                        sim.model.actuator_name2id('A_' + column_name)]

            if useexcitationcontrol and optimize_excitations:
                for frame_skip_index in range(simulation_frame_skip):
                    if not use_qacc:
                        forward_qvel_testing = sim.data.qvel.copy()
                    activations, activations_first_derivative = muscle_activation_model_secondorder(
                        activations, activations_first_derivative, ctrl,
                        sim.model.opt.timestep)

                    sim.data.ctrl[:] = activations
                    sim.step()
                    if (frame_skip_index + 1) % frame_skip_data == 0:
                        for column_name in physical_joints + virtual_joints:
                            output_table.loc[
                                next_index + frame_skip_index // frame_skip_data, column_name + '_pos'] = \
                                sim.data.qpos[
                                    sim.model.joint_name2id(column_name)]
                            output_table.loc[
                                next_index + frame_skip_index // frame_skip_data, column_name + '_vel'] = \
                                sim.data.qvel[
                                    sim.model.joint_name2id(column_name)]
                            if use_qacc:
                                output_table.loc[
                                    next_index + frame_skip_index // frame_skip_data, column_name + '_acc'] = \
                                    sim.data.qacc[
                                        sim.model.joint_name2id(column_name)]
                            else:
                                output_table.loc[
                                    next_index + frame_skip_index // frame_skip_data, column_name + '_acc'] = (
                                                                                                                      sim.data.qvel[
                                                                                                                          sim.model.joint_name2id(
                                                                                                                              column_name)] -
                                                                                                                      forward_qvel_testing[
                                                                                                                          sim.model.joint_name2id(
                                                                                                                              column_name)]) / sim.model.opt.timestep

            else:
                for frame_skip_index in range(simulation_frame_skip):
                    if not use_qacc:
                        forward_qvel_testing = sim.data.qvel.copy()
                    sim.data.ctrl[:] = ctrl
                    sim.step()
                    if (frame_skip_index + 1) % frame_skip_data == 0:
                        for column_name in physical_joints + virtual_joints:
                            output_table.loc[
                                next_index + frame_skip_index // frame_skip_data, column_name + '_pos'] = \
                                sim.data.qpos[
                                    sim.model.joint_name2id(column_name)]
                            output_table.loc[
                                next_index + frame_skip_index // frame_skip_data, column_name + '_vel'] = \
                                sim.data.qvel[
                                    sim.model.joint_name2id(column_name)]
                            if use_qacc:
                                output_table.loc[
                                    next_index + frame_skip_index // frame_skip_data, column_name + '_acc'] = \
                                    sim.data.qacc[
                                        sim.model.joint_name2id(column_name)]
                            else:
                                output_table.loc[
                                    next_index + frame_skip_index // frame_skip_data, column_name + '_acc'] = (
                                                                                                                      sim.data.qvel[
                                                                                                                          sim.model.joint_name2id(
                                                                                                                              column_name)] -
                                                                                                                      forward_qvel_testing[
                                                                                                                          sim.model.joint_name2id(
                                                                                                                              column_name)]) / sim.model.opt.timestep

            if store_mj_inverse:
                mujoco_py.cymj._mj_inverse(sim.model, sim.data)
                for column_name in physical_joints + virtual_joints:
                    output_table.loc[index, 'ID_' + column_name] = sim.data.qfrc_inverse[
                        sim.model.joint_name2id(column_name)]

            opt_error_pos = np.linalg.norm(sim.data.qpos[physical_joint_ids] -
                                           next_qpos_data[int(timestep / timestep_data) - 1][
                                               physical_joint_ids])
            opt_error_vel = np.linalg.norm(sim.data.qvel[physical_joint_ids] -
                                           next_qvel_data[int(timestep / timestep_data) - 1][
                                               physical_joint_ids])
            if use_qacc:
                opt_error_acc = np.linalg.norm(sim.data.qacc[physical_joint_ids] -
                                               next_qacc_data[int(timestep / timestep_data) - 1][
                                                   physical_joint_ids])
            else:
                opt_error_acc = np.linalg.norm(((sim.data.qvel[physical_joint_ids] -
                                                 forward_qvel_testing[
                                                     physical_joint_ids]) / sim.model.opt.timestep)
                                               - next_qacc_data[int(timestep / timestep_data) - 1][
                                                   physical_joint_ids])
            output_table.loc[index, "opt_success"] = feasible_controls_sol.success
            output_table.loc[index, "opt_error"] = feasible_controls_sol.fun
            output_table.loc[index, "opt_error_pos"] = opt_error_pos
            output_table.loc[index, "opt_error_vel"] = opt_error_vel
            output_table.loc[index, "opt_error_acc"] = opt_error_acc
            if opt_error_pos > 5 / 180 * np.pi:  # raise warning if joint angle error is larger than 5 degrees
                print(f"WARNING: Joint angle error is too large ({opt_error_pos} radians)!")
            print(f"POS: {opt_error_pos}")
            print(f"VEL: {opt_error_vel}")
            print(f"ACC: {opt_error_acc}")

        if index % 1000 == 0:
            print('{} / {}'.format(index, trajectories_table.shape[0]))

    # if second-order muscle dynamics are used, compute activation derivatives and actual control
    if useexcitationcontrol and not optimize_excitations:
        for column_name in physical_joints:
            output_table.loc[:, 'Adot_' + column_name] = np.array([np.nan] * output_table.shape[0])
            output_table.loc[:, 'ctrl_' + column_name] = np.array([np.nan] * output_table.shape[0])
            output_table.iloc[::int(timestep / timestep_data),
            output_table.columns.get_loc('Adot_' + column_name)] = np.concatenate(((output_table.iloc[
                                                                                    ::int(timestep / timestep_data),
                                                                                    output_table.columns.get_loc(
                                                                                        'A_' + column_name)].values[
                                                                                    1:] -
                                                                                    output_table.iloc[
                                                                                    ::int(timestep / timestep_data),
                                                                                    output_table.columns.get_loc(
                                                                                        'A_' + column_name)].values[
                                                                                    :-1]) / timestep, [np.nan]))
            output_table.iloc[::int(timestep / timestep_data),
            output_table.columns.get_loc('ctrl_' + column_name)] = np.concatenate(
                (muscle_activation_model_secondorder_inverse(
                    output_table.iloc[::int(timestep / timestep_data),
                    output_table.columns.get_loc('A_' + column_name)].values,
                    output_table.iloc[::int(timestep / timestep_data),
                    output_table.columns.get_loc('Adot_' + column_name)].values,
                    timestep), [np.nan]))

        # Testing forward-backward compatibility of 'muscle_activation_model_secondorder_inverse()' and 'muscle_activation_model_secondorder()':
        activation_fwd_cp_new = np.array(
            [output_table['A_' + column_name][0] for column_name in physical_joints])
        activation_first_derivative_fwd_cp_new = np.array(
            [output_table['Adot_' + column_name][0] for column_name in physical_joints])
        activation_fwd_cp = [list(activation_fwd_cp_new)]
        activation_first_derivative_fwd_cp = [list(activation_first_derivative_fwd_cp_new)]
        for i in range(0, output_table['ctrl_' + column_name].shape[0] - 1, int(timestep / timestep_data)):
            activation_fwd_cp_new, activation_first_derivative_fwd_cp_new = muscle_activation_model_secondorder(
                activation_fwd_cp_new,
                activation_first_derivative_fwd_cp_new,
                np.array([output_table['ctrl_' + column_name][i] for column_name in
                          physical_joints]),
                timestep)
            activation_fwd_cp += [list(activation_fwd_cp_new)]
            activation_first_derivative_fwd_cp += [list(activation_first_derivative_fwd_cp_new)]
        for i, column_name in enumerate(physical_joints):
            output_table.loc[:, 'A_' + column_name + '_fwd_cp'] = np.array([np.nan] * output_table.shape[0])
            output_table.loc[:, 'Adot_' + column_name + '_fwd_cp'] = np.array([np.nan] * output_table.shape[0])
            output_table.iloc[::int(timestep / timestep_data),
            output_table.columns.get_loc('A_' + column_name + '_fwd_cp')] = [activation_current[i] for
                                                                             activation_current in activation_fwd_cp]
            output_table.iloc[::int(timestep / timestep_data),
            output_table.columns.get_loc('Adot_' + column_name + '_fwd_cp')] = [activation_first_derivative_current[i]
                                                                                for activation_first_derivative_current
                                                                                in activation_first_derivative_fwd_cp]

    # Reset some environment properties to continue with regular training:
    # param_init_qpos = None  #TODO: reset to original init values
    # param_init_qvel = None
    # simulator.skip_target_choice = False

    if store_results_to_csv:
        store_trajectories_table(output_table_filename, output_table, unique_filename=False)

    return output_table


def compute_gears_and_ctrlranges(DIRNAME_CFAT,
              use_MAD_outliers=False,
              use_3xSTD_outliers=True,
              MAD_criterion_coefficient=20,
              lower_quantile=0.0005  # only used if both use_MAD_outliers and use_3xSTD_outliers are False
              ):
    """
    Compute gears and control ranges from CFAT data.
    :param DIRNAME_CFAT: directory with CFAT files (all *_CFAT.csv files in this directory are used)
    :param use_MAD_outliers: whether to use the "median absolute deviation" outlier criterion
    :param use_3xSTD_outliers: whether to use the "3x standard deviation" outlier criterion
    :param MAD_criterion_coefficient: "median absolute deviation" scaling coefficient
    :param lower_quantile: lower quantile used if both use_MAD_outliers and use_3xSTD_outliers are False
    (as float; lower_quantile=0 corresponds to min/max)
    :return: Dictionary containing a tuple of shape (gear, lower control range bound, upper control range bound)
    for each joint. The value for a "*_fwd_cp" entry should match the value for the respective joint
    (double check of forward-backward compatibility of CFAT_algorithm(optimize_excitations=False)).
    """
    upper_quantile = 1 - lower_quantile

    CFAT_table = pd.DataFrame()
    CFAT_files = os.path.expanduser(os.path.join(DIRNAME_CFAT, f'*_CFC.csv'))
    for file in glob.iglob(CFAT_files, recursive=True):
        df = pd.read_csv(file, index_col="time")
        if "opt_success" not in df:
            logging.warning(f"Skip file {file}, as it does not have the right structure.")
            continue
        CFAT_table = pd.concat((CFAT_table, df))

    # Remove samples that did not converge
    if CFAT_table.loc[CFAT_table.opt_success == False].shape[0] > 0:
        print(
            f'INFO: {CFAT_table.loc[CFAT_table.opt_success == False].shape[0]} sample{"s" * (CFAT_table.loc[CFAT_table.opt_success == False].shape[0] != 1)} did not converge!')
    CFAT_table = CFAT_table.loc[CFAT_table.opt_success != False]
    control_columns_prefix = "A_"  # "ctrl_"
    computed_feasible_controls = CFAT_table.loc[:, [column_name for column_name in CFAT_table.columns if
                                                    column_name.startswith(control_columns_prefix)]]

    participant_torque_dict = {}

    if use_MAD_outliers:
        logging.info(f"Gears and control limits using outlier criterion based on {MAD_criterion_coefficient} times MAD:")
        for column_name in computed_feasible_controls.columns:
            computed_feasible_controls__current_joint_without_outliers = \
                computed_feasible_controls[column_name].loc[np.abs(
                    computed_feasible_controls[column_name] - computed_feasible_controls[
                        column_name].median()) <= MAD_criterion_coefficient * np.abs(
                    computed_feasible_controls[column_name] - computed_feasible_controls[
                        column_name].median()).median()]
            torque_range = [computed_feasible_controls__current_joint_without_outliers.min(),
                            computed_feasible_controls__current_joint_without_outliers.max()]
            new_gear = np.max(np.abs(torque_range))
            participant_torque_dict[column_name[len(control_columns_prefix):]] = (
                new_gear, torque_range[0] / new_gear, torque_range[1] / new_gear)
    elif use_3xSTD_outliers:
        logging.info(f"Gears and control limits using outlier criterion based on 3 times STD:")
        for column_name in computed_feasible_controls.columns:
            computed_feasible_controls__current_joint_without_outliers = \
                computed_feasible_controls[column_name].loc[
                    (np.abs(stats.zscore(computed_feasible_controls[column_name], nan_policy="omit")) <= 3)]
            torque_range = [computed_feasible_controls__current_joint_without_outliers.min(),
                            computed_feasible_controls__current_joint_without_outliers.max()]
            new_gear = np.max(np.abs(torque_range))
            participant_torque_dict[column_name[len(control_columns_prefix):]] = (
                new_gear, torque_range[0] / new_gear, torque_range[1] / new_gear)
    else:
        logging.info(f"Gears and control limits using {lower_quantile * 100}% and {upper_quantile * 100}% quantiles:")
        for column_name in computed_feasible_controls.columns:
            torque_range = [computed_feasible_controls[column_name].quantile(q=lower_quantile),
                            computed_feasible_controls[column_name].quantile(q=upper_quantile)]
            new_gear = np.max(np.abs(torque_range))
            participant_torque_dict[column_name[len(control_columns_prefix):]] = (
                new_gear, torque_range[0] / new_gear, torque_range[1] / new_gear)

    return participant_torque_dict
