# Computation of Feasible Applied Torques (CFAT)

This python package allows to compute feasible applied torques (i.e., sequences of rotational forces directly acting on the joints) from given Inverse Kinematics data (e.g., obtained from a MoCap user study) for a specific torque-driven MuJoCo model. In addition, these computed torque sequences can be used to identify appropriate gears and control ranges for this MuJoCo model.
For details, see the paper "Simulating Interaction Movements via Model Predictive Control".

### Example scripts:
- **iso_vr_pointing** (ISO-VR-Pointing Dataset from "Simulating Interaction Movements via Model Predictive Control")
  - [`iso_vr_pointing_example.py`](https://github.com/fl0fischer/cfat/blob/main/examples/iso_vr_pointing/iso_vr_pointing_example.py): run CFAT on a single Inverse Kinematics file
  - [`iso_vr_pointing_complete.py`](https://github.com/fl0fischer/cfat/blob/main/examples/iso_vr_pointing/iso_vr_pointing_complete.py): run CFAT on all Inverse Kinematics files sequentially
  - [`iso_vr_pointing_complete_slurm.py`](https://github.com/fl0fischer/cfat/blob/main/examples/iso_vr_pointing/iso_vr_pointing_complete_slurm.py): run CFAT on all Inverse Kinematics files on a cluster using SLURM
  - [`iso_vr_pointing_gears.py`](https://github.com/fl0fischer/cfat/blob/main/examples/iso_vr_pointing/iso_vr_pointing_gears.py): compute model gears and control ranges from the identified CFAT data

### Installation:
- To install with [mujoco-py](https://github.com/openai/mujoco-py), run
```bash
pip install -e .
```
or 
```bash
pip install -e .["mujoco_py"]
```
- To install with [mujoco](https://github.com/deepmind/mujoco), run
```bash
pip install -e .["mujoco"]
```
