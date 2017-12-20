# neural-dependency-parser
Greedy Transition-based dependency parsers based on neural network. 

### Note:
* Main techniques behind the parser are from the paper [A Fast and Accurate Dependency Parser using Neural Networks](http://cs.stanford.edu/people/danqi/papers/emnlp2014.pdf)
* The repo is based on the starter codes from cs224n assigment2. [here](http://web.stanford.edu/class/cs224n/assignment2/assignment2.zip)
* Current best UAS in dev set is 87.95.

### Train:
* Please check the cs224n [assignment 2](http://web.stanford.edu/class/cs224n/assignment2/index.html) for installing dependencies etc.
* Set the hyper-parameters in `config.py`.
* Then run with existing dataset
```shell
python train.py
```


### TODO
* add reg term
* add decay learning rate
* add extra hidden layer
* add POS & LABEL embeddings
* add pre-trained glove wordvector init
* add tensorboard