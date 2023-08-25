# Peg-in-Hole with RL algorithm

This project provides diff models to complete the assembly tasks (peg-in-hole) in soft body and rigid body environmnets. The RL-PPO algorithm is mainly used in this project and the task for rigid body environment is completed (the policy includes hole seaching, shape matching, insertion). For the soft part, it uses soft body (soft tool) to complete the insertion task.

Details and demos: [https://0707yiliu.github.io/peg-in-hole-with-RL/](https://0707yiliu.github.io/peg-in-hole-with-RL/)

## Description

This repo contains the whole RL framework but without the trained model. Running code in ./src, RL-env in gym_envs. You can use the command in ./src/command.txt to quick start a trainning.

## Quick Started
* Register the gym-env (go to the repo root)
```
pip install -e .
```
* quick trainning
```
python /your/repo/root/src/run.py  -e URPiHDense-v1 -t 7000000 -a PPO -v vision-touch -lr 0.0003 -hs 4mm -l -dsl -dr -nor -g 3
```

### Dependencies

* gym >= 2.1
* Mujoco >= 2.3
* Python >= 3.7
* KDL-pkg
* ...

<!-- # Rigid body assembly

This sub-work build up the env with triangle obj and hole, the PPO-algorithm is used to train the model with different gaps (we used the gap proportion the redefine the gap between obj & hole finally).

Performance:[https://www.youtube.com/watch?v=1npPWYU3B6g](https://www.youtube.com/watch?v=1npPWYU3B6g)

# Soft body insertion

TODO -->

<!-- 
### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
``` -->

<!-- ## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
``` -->

<!-- ## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie) -->

## Version History

* 0.2
    * Update soft body version
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

Distributed under the MIT License.
