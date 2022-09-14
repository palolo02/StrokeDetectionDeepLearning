# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# # Parameters and DataLoaders
# input_size = 5
# output_size = 2

# batch_size = 30
# data_size = 100

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class RandomDataset(Dataset):

#     def __init__(self, size, length):
#         self.len = length
#         self.data = torch.randn(length, size)

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return self.len

# rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
#                          batch_size=batch_size, shuffle=True)

# class Model(nn.Module):
#     # Our model

#     def __init__(self, input_size, output_size):
#         super(Model, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)

#     def forward(self, input):
#         output = self.fc(input)
#         print("\tIn Model: input size", input.size(),
#               "output size", output.size())

#         return output

# model = Model(input_size, output_size)
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = nn.DataParallel(model)
# else:
#     print("Let's use 1 GPU!")
  

# model.to(device)          

# for data in rand_loader:
#     input = data.to(device)
#     output = model(input)
#     print("Outside: input size", input.size(),
#           "output_size", output.size())



import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():    
    dist.init_process_group("nccl")
    #rank = dist.get_rank()
    #print(f"Start running basic DDP example on rank {rank}.")

    # # create model and move it to GPU with id rank
    # device_id = rank % torch.cuda.device_count()
    # model = ToyModel().to(device_id)
    # ddp_model = DDP(model, device_ids=[device_id])

    # loss_fn = nn.MSELoss()
    # optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # optimizer.zero_grad()
    # outputs = ddp_model(torch.randn(20, 10))
    # labels = torch.randn(20, 5).to(device_id)
    # loss_fn(outputs, labels).backward()
    # optimizer.step()

if __name__ == "__main__":
    demo_basic()