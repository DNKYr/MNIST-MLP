# Error Report
> This is a report creating after I intentionally break a MNIST classfier's traning process. 

I have done the following action to break it: 
- Bad Learning Rate: Learning rate too big/small
- random label: Random label to training image
- remove ReLU from network: Make it a linear transformation
- remove zero_grad: make the gradient descent impossible

Here is the result of each: 
## Small Learning rate
Loss barely move, because I can't desecnd on the field
## Big Learning Rate
Loss first drastically increase, than stable around it: the neural network jump between the hills of the minimum
## Random Label 
For large sample and few epoch, the loss doesn't change at all. The loss stabilize around 2.30 because that is the value when a model is guessing randomly. For fewer sample and more epoch, the training loss start descend, while the test loss start increase. At the end, training loss is at 0.001 while the test loss is at 9.235. The train accuracy goes up while the test accuracy stays around 10%

This is because of the memoization in neural net's learning: the neural net stored the characteristic of each picture of the training data on a single weight. However, this memoization is useless because the model has just remembered 100% of noice 

## Remove ReLU
For this one, the accuracy stabilized around 91.4% after 5 epoch. This is because without ReLU, the neural net can be just sum down into a single linear transformation matrix: $\left[ 784 \times 10 \right]$ by matrix multiplication. Here is the math for such a neural net without ReLU

$$
\begin{aligned}
y &= W_3\big(W_2(W_1x + b_1)+b_2\big)+b_3 \\
&= W_3(W_2W_1x + W_2b_1 + b_2) + b_3 \\
&= W_3W_2W_1x + W_3W_2b_1+W_3b_2+b_3 \\
&= W'X + b'
\end{aligned}
$$
where
$$
\underbrace{W'}_{10 \times784} = \underbrace{W_3}_{10 \times 128} \,  \underbrace{W_2}_{128\times128} \, \underbrace{W_1}_{128 \times 784}, 
\qquad
b'=W_3W_2b_1+W_3b_2+b_3
$$
Without a nonlinearity, any stack of linear layers is equivalent to a single linear layer, so adding depth adds no expressive power. ReLU between layers prevents this collapse, because $W_2,\mathrm{ReLU}(W_1x + b_1)$ can't be rewritten as a single matrix multiplication.

## Remove zero_grad
Descend intially. But failed later as the current correct gradient got mixed up with the previous one. So the gradient is going to the wrong direction. 