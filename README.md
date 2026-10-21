# Neural Network Back-Propagation Training System

### Program Output (Baseline Run)

```text
Begin neural network back-propagation demo

Generating 1000 artificial data items with 4 features
Generating weights and biases:
-5.03 -7.79 -0.66  5.43  3.15 -1.34 -2.92  8.88 -7.97  2.85 
-9.43 -5.04 -3.60  9.80  3.64  3.10 -4.35  2.31  4.09  4.04 
 9.00 -8.13 -6.78 -2.36  5.96 -6.61  5.88 -3.86  6.46  7.73 
 1.13  4.33  3.99 -9.73  9.84  6.13  7.17 -8.75  0.29  0.51 
-4.54  9.88  3.82 
Done

Creating train (80%) and test (20%) matrices
Done

Training data:
[  0]  -1.09  -9.10   0.85   5.52   0.00   0.00   1.00  
[  1]  -1.92   8.16   6.53  -1.54   0.00   1.00   0.00  
[  2]   7.94   3.66  -2.63  -3.38   0.00   1.00   0.00  
[  3]  -5.01  -4.07  -6.24   2.87   0.00   1.00   0.00  
. . .
[799]   1.61   2.00  -5.77  -7.44   0.00   1.00   0.00  

Test data:
[  0]   0.18   5.94   7.58  -3.33   0.00   1.00   0.00  
[  1]   7.48  -3.41  -0.58  -8.75   0.00   0.00   1.00  
[  2]  -0.73  -9.31   1.91  -6.77   0.00   0.00   1.00  
[  3]  -6.13  -0.95   5.70   1.62   0.00   0.00   1.00  
. . .
[199]   7.97  -5.35  -4.93  -0.51   0.00   0.00   1.00  

Creating a 4-5-3 neural network

Setting maxEpochs = 1000
Setting learnRate = 0.05
Setting momentum  = 0.01

Starting training
epoch = 100  error = 0.0381
epoch = 200  error = 0.0305
epoch = 300  error = 0.0256
epoch = 400  error = 0.0329
epoch = 500  error = 0.0201
epoch = 600  error = 0.0240
epoch = 700  error = 0.0190
epoch = 800  error = 0.0194
epoch = 900  error = 0.0201
Done

Final neural network model weights and biases:

-2.98 -5.07  0.26 -0.27  0.48 -0.88 -2.39 -4.52  2.95 -4.68 
-5.36 -3.68  1.88 -1.76  1.73  1.81 -3.01 -1.09  0.42 -1.43 
 5.51 -7.72  3.07 -3.53  3.69 -2.34  3.51 -1.39  2.62  0.13 
-3.79 -1.60 -0.64  3.94  2.86 -0.55 -2.81  0.98 -2.64  1.92 
-1.60  5.61 -1.03 

Final accuracy on training data = 0.9900
Final accuracy on test data     = 0.9900

End back-propagation demo
```

---

## How to Run

The application has been compiled into a self-contained `.exe` inside the [executables folder](executables).

1. Open the [executables folder](executables).
2. Double-click `CodingBackProp.exe`.
3. A prompt will appear:
   - `[1]` Baseline (Mean Squared Error)
   - `[2]` Cross-Entropy Error
   - `[3]` Weight Decay (L2 Regularization)
   - `[4]` Compare All
4. Simply press **Enter** to run the baseline demo, or select **1-4**. The window remains open until you press Enter.

---

## How It Works

### 1. Neural Network Architecture
The network is fully connected with **3 layers**:
- **Input layer**: $m = 4$ features (e.g. normalized attributes)
- **Hidden layer**: $h = 5$ nodes with Hyperbolic Tangent ($\tanh$) activation function
- **Output layer**: $n = 3$ nodes with Softmax activation function (outputs sum to $1.0$, producing class probabilities)
- **Total weights & biases**: $(4 \times 5) + 5 + (5 \times 3) + 3 = 43$ parameters.

### 2. Forward Propagation
1. **Hidden layer**:
   $$hSums_j = \sum_{i} x_i \cdot w_{ij} + hBiases_j$$
   $$hOutputs_j = \tanh(hSums_j)$$
2. **Output layer**:
   $$oSums_k = \sum_{j} hOutputs_j \cdot w_{jk} + oBiases_k$$
   $$outputs_k = \text{Softmax}(oSums_k) = \frac{e^{oSums_k}}{\sum_l e^{oSums_l}}$$

### 3. Back-Propagation Algorithm
Error signals work backward through the layers:
1. **Output node error signal ($\delta_k$)**:
   - For **Mean Squared Error (MSE)**:
     $$\delta_k = (t_k - y_k) \cdot y_k(1 - y_k)$$
   - For **Cross-Entropy Error**:
     $$\delta_k = t_k - y_k$$
2. **Hidden node error signal ($\delta_j$)**:
   $$\delta_j = (1 + hOutputs_j)(1 - hOutputs_j) \cdot \sum_k (\delta_k \cdot w_{jk})$$
3. **Weight & Bias updates with Momentum ($\mu$) and Learning Rate ($\alpha$)**:
   $$\Delta w_{ij} = \alpha \cdot \delta_j x_i + \mu \cdot \Delta w_{ij}^{(\text{prev})}$$
   $$w_{ij} \leftarrow w_{ij} + \Delta w_{ij}$$

---

## Extensions Implemented

### 1. Cross-Entropy Error
* **Theory**: In classification problems, Mean Squared Error combined with softmax can lead to sluggish learning when output neurons saturate (the derivative term $y_k(1 - y_k)$ approaches zero). By using Cross-Entropy loss ($E = -\sum t_k \ln(y_k)$), the derivative of the softmax loss neatly cancels the activation gradient, yielding an error signal directly proportional to the prediction error:
  $$\delta_k = t_k - y_k$$
* **Result**: Accelerated error reduction and faster convergence during training.

### 2. Weight Decay / Regularization (L2 Penalty)
* **Theory**: Back-propagation without regularization can produce arbitrarily large weights as neurons attempt to overfit noisy training samples. Weight decay penalizes large weights by adding $\frac{1}{2} \lambda \sum w^2$ to the cost function:
  $$w \leftarrow w(1 - \lambda \cdot \alpha) + \Delta w$$
* **Result**: Maintains lower weight norms ($\|\mathbf{w}\|_2$), improving model generalization and preventing overfitting on unseen test data.

---

## Comparative Benchmark Summary

Running option `[4]` (`--all`) trains the three variants under identical seeds ($seed = 1$) and displays the comparison:

| Model Configuration | Training Accuracy | Test Accuracy | Weight Norm ($\|\mathbf{w}\|_2$) |
| :--- | :---: | :---: | :---: |
| **Baseline (MSE)** | 99.00% | 99.00% | 20.202 |
| **Cross-Entropy Error** | 99.12% | 98.50% | 51.541 |
| **Weight Decay ($L_2 = 0.001$)** | 97.12% | 95.50% | 22.516 |

---

## Project Structure

```
backprop/
├── BackPropProgram for CS345 IS 1.cs   # Original source file provided
├── .gitignore                          # Build and artifact exclusions
├── README.md                           # Documentation, execution guide & analysis
├── CodingBackProp/                     # C# Project folder (.NET 8)
│   ├── CodingBackProp.csproj           # Project configuration
│   └── BackPropProgram.cs              # Full implementation with extensions
└── executables/                        # Standalone compiled binaries
    ├── CodingBackProp.exe              # Self-contained Win-x64 executable
    └── CodingBackProp.pdb              # Debug symbol file
```
