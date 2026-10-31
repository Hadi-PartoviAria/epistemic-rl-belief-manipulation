# Epistemic Reinforcement Learning for Belief Manipulation: SAT-Based Reasoning with Reward Machines

Code for the paper by Hadi Partovi Aria, Zhe Xu. IEEE Control Systems Letters, 2026.

This repository has the code for the three case studies in the paper. In each, the agent's observations include an observer's beliefs, tracked with epistemic logic, and the objective is written as a reward machine. Implicit beliefs (what follows from the explicit belief base under common ground) are computed with a SAT solver.

## Layout

`chop_cup/` is Case Study 1, the cups-and-balls (chop cup) task. `chop_cup_env.py` defines the environment, `epistemic_logic.py` the belief base and SAT entailment check, and `reward_machine.py` the reward machine. The learners are `qrm.py` (Q-learning with reward machines), `qrm_pbrs.py` (QRM with potential-based reward shaping), `hrl_agent.py` and `hrl_qrm.py` (hierarchical variants) and `ddqn_agent.py` (Double DQN baseline). `run_comparison.py` runs all methods and plots the comparison. `tutorial.py` walks through the components step by step.

`three_card_monte/` is Case Study 2, the three-card monte task, with the same set of agents and `three_card_env.py` as the environment.

`autonomous_driving/` is Case Study 3, Waymo-style driving on a 7x7 grid. Puddles and floods look alike, as do construction cones and closed lanes, so the visual label alone does not say whether a cell is drivable. The observer combines it with traffic evidence (whether other cars pass or avoid the cell), and the SAT-derived implicit belief gives the drivability verdict. `run_comparison.py` compares four agents: BEQRM, BEQRM with potential-based reward shaping, HRL-RM and DDQN.

`autonomous_driving_ablation/` is the ablation on the value of the implicit belief. `ablation_agent.py` defines five variants that switch the explicit belief, the implicit belief, the reward machine state and reward shaping on or off, and `run_ablation.py` runs them on the same fixed grid.

## Running

```bash
pip install -r requirements.txt
cd chop_cup && python run_comparison.py
cd ../three_card_monte && python run_comparison.py
cd ../autonomous_driving && python run_comparison.py
cd ../autonomous_driving_ablation && python run_ablation.py
```

## Related code

The epistemic RL formulation follows Engesser et al., "A Simple Integration of Epistemic Logic and Reinforcement Learning" (AAMAS 2025), whose Rust implementation is at https://github.com/tengesser/epirl. That code is not included here.

## Citation

```bibtex
@article{aria2026epistemic,
  title = {Epistemic Reinforcement Learning for Belief Manipulation: SAT-Based Reasoning with Reward Machines},
  author = {Hadi Partovi Aria and Zhe Xu},
  journal = {IEEE Control Systems Letters},
  year = {2026}
}
```

## License

MIT, see `LICENSE`.
