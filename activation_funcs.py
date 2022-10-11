import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# fig, ax = plt.subplots(2,4)
input_data = torch.arange(-10,10,0.2)


# Rectified linear unit (ReLU)
act = nn.ReLU()
output = act(input_data)
# ax[0,0].plot(input_data, output, label="ReLU")
# ax[0,0].grid()
# ax[0,0].legend()
plt.clf()
plt.plot(input_data, output, label="ReLU")
plt.grid()
plt.legend()
plt.savefig("relu.pdf", format="pdf", bbox_inches="tight")

# Leaky ReLU
slope = 0.2
act = nn.LeakyReLU(slope)
output = act(input_data)
# ax[0,1].plot(input_data, output, label="Leaky ReLU")
# ax[0,1].grid()
# ax[0,1].legend()
plt.clf()
plt.plot(input_data, output, label="Leaky ReLU")
plt.grid()
plt.legend()
plt.savefig("leakyRelu.pdf", format="pdf", bbox_inches="tight")

# Gaussian error linear unit (GELU)
act = nn.GELU()  # You can use a tanh approximation passing the argument approximate='tanh'
output = act(input_data)
# ax[0,2].plot(input_data, output, label="GELU")
# ax[0,2].grid()
# ax[0,2].legend()
plt.clf()
plt.plot(input_data, output, label="GELU")
plt.grid()
plt.legend()
plt.savefig("gelu.pdf", format="pdf", bbox_inches="tight")

# Exponential Linear Unit (ELU)
act = nn.ELU()
output = act(input_data)
# ax[0,3].plot(input_data, output, label="ELU")
# ax[0,3].grid()
# ax[0,3].legend()
plt.clf()
plt.plot(input_data, output, label="ELU")
plt.grid()
plt.legend()
plt.savefig("elu.pdf", format="pdf", bbox_inches="tight")

# Sigmoid Linear activation unit (SiLU)
act = nn.Sigmoid()
output = act(input_data)
# ax[1,0].plot(input_data, output, label="Sigmoid")
# ax[1,0].grid()
# ax[1,0].legend()
plt.clf()
plt.plot(input_data, output, label="SiLU")
plt.grid()
plt.legend()
plt.savefig("silu.pdf", format="pdf", bbox_inches="tight")

# Hyperbolic tangent (tanh)
act = nn.Tanh()
output = act(input_data)
# ax[1,1].plot(input_data, output, label="Tanh")
# ax[1,1].grid()
# ax[1,1].legend()
plt.clf()
plt.plot(input_data, output, label="tanh")
plt.grid()
plt.legend()
plt.savefig("tanh.pdf", format="pdf", bbox_inches="tight")

# Softmax function 
act = nn.Softmax(dim=0)
output = act(input_data)
# ax[1,2].plot(input_data, output, label="Softmax")
# ax[1,2].grid()
# ax[1,2].legend()
# print(f"Sum of all softmax output is {torch.sum(output)}")
plt.clf()
plt.plot(input_data, output, label="softmax")
plt.grid()
plt.legend()
plt.savefig("softmax.pdf", format="pdf", bbox_inches="tight")

# Softplus
act = nn.Softplus()
output = act(input_data)
# ax[1,3].plot(input_data, output, label="Softplus")
# ax[1,3].grid()
# ax[1,3].legend()
plt.clf()
plt.plot(input_data, output, label="softplus")
plt.grid()
plt.legend()
plt.savefig("softplus.pdf", format="pdf", bbox_inches="tight")

# plt.show()
# plt.savefig("relu.pdf", format="pdf", bbox_inches="tight")
plt.grid()
plt.legend()


