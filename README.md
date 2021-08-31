# Logistic Bandit Experiments 

## Run regret minimization exps 
The file _regret_minimization.py_ allows you to run simple distributed experiments with some base algorithms. 

To change hyper-parameters (horizon, algo, ..), just change the file _configs/example_config.json_. I'll implement some additional functions to automate the launch of experiments and their post-process soon. 

## Others

The files for misc. experiments (_e.g_ try out the data-dependent testing for the online procedure) are in the directory _misc/_. (**not finished yet**)
What's implemented so far is the online learning procedure (with preconditioning for fast solving and efficient ellipsoidal projection)


