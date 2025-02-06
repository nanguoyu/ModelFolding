import matplotlib.pyplot as plt

# Given inertia values
inertia_dict = {
    "relu_conv": 39.86,
    "layer1.0.1": 8.28,
    "layer1.1.1": 8.14,
    "layer2.0.1": 24.73,
    "layer2.0.2": 80.65,
    "layer2.1.1": 24.07,
    "layer3.0.1": 52.67,
    "layer3.0.2": 88.76,
    "layer3.1.1": 29.74,
    "layer4.0.1": 9.24,
    "layer4.0.2": 6.08,
    "layer4.1.1": 1.22,
}

# Extracting layers and inertia values
layers = list(inertia_dict.keys())
inertia_values = list(inertia_dict.values())

# Plotting
plt.figure(figsize=(14, 6))  # Increase figure width
plt.bar(layers, inertia_values, color='steelblue')
plt.xlabel("Layer", fontsize=18)
plt.ylabel("Inertia Value", fontsize=18)
plt.title("Layer-wise Inertia Values for Weight Clustering", fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()
plt.savefig("resnet18_cifar10_split_plot.png")