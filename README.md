## Multi-Agent Distributed Reinforcement Learning for Making Decentralized Offloading Decisions

### Background
 This is the source code for a paper submission.

### Requirements

 pandas==0.24.2

 matplotlib==3.1.2

 numpy==1.16.4

 scipy==1.2.1

 seaborn==0.9.0

 torch==1.4.0

 Mesa==0.8.6

### Install the package

 Go to the virtual environment and the root `v2x` folder, run:

```console
foo@bar:~/v2x$ python -m pip install -r requirements.txt
foo@bar:~/v2x$ python -m pip install -e .
```

### Generate the performance charts in paper

 extract all files in `logs` folder (use "extract here"), create a new `graphs` folder, and run:

```console
foo@bar:~/v2x$ python output.py
```

 The charts will be created in the `graphs` folder.

### Run the code in predefined modes

Create a new `models` folder, the models will be saved here.

1. To run the DRACO algorithm in training mode:

	```console
	foo@bar:~/v2x$ python run.py draco
	```

2. To permit rebidding,add keyword "rebid":

	```console
	foo@bar:~/v2x$ python run.py draco_rebid
	```

	Maximum permitted rebidding is specified in the config file in `v2x/config` folder, parameter `nrRebid`.

3. To run in evaluation mode, add keyword "eval":

	```console
	foo@bar:~/v2x$ python run.py draco_rebid_eval
	```

 	Performance output will be automatically created in the `logs` folder. If there are existing files in the folder with the same name, the new results will be appended to the end.

4. To run benchmark RIAL:

	```console
	foo@bar:~/v2x$ python run.py rial
	```

5. To permit rebidding for RIAL, add keyword "rebid":

	```console
	foo@bar:~/v2x$ python run.py rial_rebid
	```


