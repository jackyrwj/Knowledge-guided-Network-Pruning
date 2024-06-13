## Knowledge-guided Network Pruning

![python](https://img.shields.io/badge/python-3.8-red)
![torch](https://img.shields.io/badge/torch-1.10.1-green)
![torcheeg](https://img.shields.io/badge/torcheeg-1.0.6-orange)
![mne](https://img.shields.io/badge/mne-1.0.3-blue)
![scipy](https://img.shields.io/badge/scipy-1.8.1-yellowgreen)
![pandas](https://img.shields.io/badge/pandas-1.4.2-brightgreen)

This is the implementation of our paper: Wenjie Rao, Sheng-hua Zhong "Knowledge-guided Network Pruning for EEG-based Emotion Recognition", accepted as a regular paper on IEEE International Conference on Bioinformatics and Biomedicine (BIBM)


> With the development of deep learning in EEG-related tasks, the complexity of learning models has gradually increased. These complex models often result in long inference times, high energy consumption, and an increased risk of overfitting. Therefore, model compression has become an important consideration. Although some EEG models have used lightweight techniques, such as separable convolution, no existing work has directly attempted to compress EEG models to reduce their complexity. In this paper, we integrate neuroscience knowledge into EEG model pruning recovery, and innovatively propose two loss functions in the learning process, the knowledge-guided region-wise loss that enforces the classification evidence consistent with the importance of the prefrontal lobe, and the knowledge-guided sample-wise loss that constrains the learning process by distinguishing the importance of different samples.

### Flow Chart

Given an EEG sample, we first train and prune a convolutional neural network. We then use guided Grad-CAM to extract classification evidence and constrain the model optimization by introducing knowledge-guided region-wise loss and sample-wise loss on data-driven evidence for better performance recovery.
![](https://raw.githubusercontent.com/jackyrwj/picb/master/rao1-p4-rao-large.gif)

### Accuracy Comparison

Results of using the combination of two proposed loss functions to recover two EEG models on DEAP. Bold font indicates higher accuracy between "Baseline" and "Proposed".

![](https://raw.githubusercontent.com/jackyrwj/picb/master/rao.t1-p4-rao-small.gif)

### Conclusion

In this paper, we investigate network pruning methods for EEG models used in emotion recognition tasks. We propose two innovative constraints in the learning process that incorporate interpretability into the recovery of pruning models. The first constraint is the knowledge-guided region-wise loss, which enforces classification evidence consistent with the importance of the prefrontal lobe. The second constraint is the knowledge-guided sample-wise loss, which distinguishes the importance of different samples to constrain the learning process. Our results show that the proposed method significantly enhances the recovery accuracy of the pruned model while reducing the required computational resources.
