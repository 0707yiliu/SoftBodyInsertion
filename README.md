# Peg-in-Hole with PPO

This project provides diff models to complete the assembly tasks (peg-in-hole) in soft body and rigid body environmnets. The RL-PPO algorithm is mainly used in this project and the task for rigid body environment is completed (the policy includes hole seaching, shape matching, insertion). For the soft part, it uses soft body (soft tool) to complete the insertion task.

Details and demos: [https://0707yiliu.github.io/peg-in-hole-with-RL/](https://0707yiliu.github.io/peg-in-hole-with-RL/)

## Description

This repo contains the whole RL framework but without the trained model. Running code in ./src

## Getting Started

### Dependencies

* Mujoco >= 2.3
* Python >= 3.7
* KDL-pkg
...

# Rigid body assembly

This sub-work build up the env with triangle obj and hole, the PPO-algorithm is used to train the model with different gaps (we used the gap proportion the redefine the gap between obj & hole finally).

Performance:[https://www.youtube.com/watch?v=1npPWYU3B6g](https://www.youtube.com/watch?v=1npPWYU3B6g)

# Soft body insertion

TODO



### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments