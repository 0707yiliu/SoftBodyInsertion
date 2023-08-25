# Peg-in-Hole with PPO

This project provides diff models to complete the assembly tasks (peg-in-hole) in soft body and rigid body environmnets. The RL-PPO algorithm is mainly used in this project and the task for rigid body environment is completed (the policy includes hole seaching, shape matching, insertion). Next step is trying to use soft body (soft tool) to complete the insertion task.

Details and demos: [https://0707yiliu.github.io/peg-in-hole-with-RL/](https://0707yiliu.github.io/peg-in-hole-with-RL/)


# Env Requstion

Mujoco >= 2.3

Python >= 3.7

KDL-pkg

...

# Rigid body assembly

This sub-work build up the env with triangle obj and hole, the PPO-algorithm is used to train the model with different gaps (we used the gap proportion the redefine the gap between obj & hole finally).

Performance:[https://www.youtube.com/watch?v=1npPWYU3B6g](https://www.youtube.com/watch?v=1npPWYU3B6g)

