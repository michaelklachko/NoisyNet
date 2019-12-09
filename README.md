# NoisyNet
Code for "Improving Noise Tolerance of Mixed-Signal Neural Networks"  https://arxiv.org/abs/1904.01705

Dataset (4 bit CIFAR-10) can be downloaded [here](https://drive.google.com/file/d/1lS_R_0pHPhUqzTpYS0C6IrtkgHfHe8cU/view?usp=sharing)

To run the model with `I_max = 1nA` in all layers (~78%):
```
python noisynet.py --current 1 --act_max 5 --w_max1 0.3 --LR 0.005 --L2_1 0.0005 --L2_2 0.0002
```
Noise-free baseline (~88%):
```
python noisynet.py --L2 0.0005 --dropout 0.1 --nepochs 450
```
