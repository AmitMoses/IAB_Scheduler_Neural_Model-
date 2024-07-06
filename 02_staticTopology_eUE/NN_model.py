import torch
import torch.nn as nn
import p_RadioParameters as rp
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class resourceAllocationNetwork(nn.Module):
    def __init__(self,
                 small_input_size=41, small_n_hidden=41, small_output_size=22,
                 big_input_size=40, big_n_hidden=40, big_output_size=10):
        super(resourceAllocationNetwork, self).__init__()
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )
        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_iabs = x[:, 0:40]
        x_ues = x[:, 40:]
        x_iabs = x_iabs.view(-1, 40)
        x_ues = x_ues.view(-1, 400)
        x_iabs = self.big(x_iabs)
        y = torch.zeros(x_iabs.shape[0], 10, 22)

        for i in range(0, 10):
            small_in_UE = x_ues[:, 0 + i * 40:40 + i * 40]
            small_in_IAB = x_iabs[:, i:i + 1]
            small_in = torch.cat((small_in_UE, small_in_IAB), dim=1)
            y[:, i, :] = self.small(small_in) * small_in_IAB

        return y


class resourceAllocation_NN_embb(nn.Module):
    def __init__(self, unit_input=4, emb_size=10,
                 small_input_size=10 * 10 + 1, small_n_hidden=101, small_output_size=22,
                 big_input_size=10 * 10, big_n_hidden=100, big_output_size=10):
        super(resourceAllocation_NN_embb, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )
        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )
        self.embbeing_size = emb_size

    def forward(self, x):
        x_iabs = x[:, 0:40]
        x_ues = x[:, 40:]

        x_ues = x_ues.view(-1, 400)
        x_iabs = torch.reshape(x_iabs, (-1, 4))
        x_iabs = self.embed(x_iabs)
        x_iabs = x_iabs.view(-1, 10 * self.embbeing_size)
        x_iabs = self.big(x_iabs)

        y = torch.zeros(x_iabs.shape[0], 10, 22)

        for i in range(0, 10):
            small_in_UE = x_ues[:, 0 + i * 40:40 + i * 40]
            small_in_UE = torch.reshape(small_in_UE, (-1, 4))
            small_in_UE = self.embed(small_in_UE)
            small_in_UE = torch.reshape(small_in_UE, (-1, 10 * self.embbeing_size))
            small_in_IAB = x_iabs[:, i:i + 1]
            small_in = torch.cat((small_in_UE, small_in_IAB), dim=1)
            y[:, i, :] = self.small(small_in) * small_in_IAB

        return y


class resourceAllocation_NN_embb_Deep(nn.Module):
    def __init__(self, unit_input=4, emb_size=10,
                 small_input_size=10 * 10 + 1, small_n_hidden=101, small_output_size=22,
                 big_input_size=10 * 10, big_n_hidden=100, big_output_size=10):
        super(resourceAllocation_NN_embb_Deep, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.Sigmoid(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.Sigmoid(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.Sigmoid(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.Sigmoid(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )
        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.Sigmoid(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.Sigmoid(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.Sigmoid(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.Sigmoid(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )
        self.embbeing_size = emb_size

    def forward(self, x):
        x_iabs = x[:, 0:40]
        x_ues = x[:, 40:]

        x_ues = x_ues.view(-1, 400)
        x_iabs = torch.reshape(x_iabs, (-1, 4))
        x_iabs = self.embed(x_iabs)
        x_iabs = x_iabs.view(-1, 10 * self.embbeing_size)
        x_iabs = self.big(x_iabs)

        y = torch.zeros(x_iabs.shape[0], 10, 22)

        for i in range(0, 10):
            small_in_UE = x_ues[:, 0 + i * 40:40 + i * 40]
            small_in_UE = torch.reshape(small_in_UE, (-1, 4))
            small_in_UE = self.embed(small_in_UE)
            small_in_UE = torch.reshape(small_in_UE, (-1, 10 * self.embbeing_size))
            small_in_IAB = x_iabs[:, i:i + 1]
            small_in = torch.cat((small_in_UE, small_in_IAB), dim=1)
            y[:, i, :] = self.small(small_in) * small_in_IAB

        return y


class resourceAllocation_NN_embb_Deep_2(nn.Module):
    def __init__(self, unit_input=4, emb_size=10,
                 small_input_size=10 * 11 + 1, small_n_hidden=101, small_output_size=24,
                 big_input_size=10 * 10, big_n_hidden=100, big_output_size=10):
        super(resourceAllocation_NN_embb_Deep_2, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )
        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )
        self.embbeing_size = emb_size

    def forward(self, x):
        x_iabs = x[:, 0:40]
        x_ues = x[:, 40:]

        x_ues = x_ues.view(-1, 440)
        x_iabs = torch.reshape(x_iabs, (-1, 4))
        x_iabs = self.embed(x_iabs)
        x_iabs = x_iabs.view(-1, 10 * self.embbeing_size)
        x_iabs = self.big(x_iabs)

        y = torch.zeros(x_iabs.shape[0], 10, 24)

        for i in range(0, 10):
            small_in_UE = x_ues[:, 0 + i * 40:40 + i * 40]
            small_in_UE = torch.reshape(small_in_UE, (-1, 4))
            small_in_UE = self.embed(small_in_UE)
            small_in_UE = torch.reshape(small_in_UE, (-1, 10 * self.embbeing_size))
            small_in_IAB = x_iabs[:, i:i + 1]
            small_in = torch.cat((small_in_UE, small_in_IAB), dim=1)
            # print(small_in.shape)
            # print(self.small(small_in).shape)
            # print(small_in_IAB.shape)
            y[:, i, :] = self.small(small_in) * small_in_IAB

        return y


# class resourceAllocation_Dynamic(nn.Module):
#     def __init__(self, unit_input=4, emb_size=10,
#                  small_input_size=10 * rp.maxUEperBS + 1, small_n_hidden=10 * rp.maxUEperBS + 1,
#                  small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2, small_donor_output_size=rp.maxUEperBS*2,
#                  big_input_size=10 * rp.IAB_num,
#                  big_n_hidden=10 * rp.IAB_num, big_output_size=10):
#         super(resourceAllocation_Dynamic, self).__init__()
#         self.embed = nn.Sequential(
#             nn.Linear(in_features=unit_input, out_features=emb_size)
#         )
#         self.big = nn.Sequential(
#             nn.Linear(big_input_size, big_n_hidden),
#             nn.ReLU(),
#             nn.Linear(big_n_hidden, big_n_hidden),
#             nn.ReLU(),
#             nn.Linear(big_n_hidden, big_n_hidden),
#             nn.ReLU(),
#             nn.Linear(big_n_hidden, big_n_hidden),
#             nn.ReLU(),
#             nn.Linear(big_n_hidden, big_output_size),
#             nn.Softmax(dim=1)
#         )
#
#         self.small = nn.Sequential(
#             nn.Linear(small_input_size, small_n_hidden),
#             nn.ReLU(),
#             nn.Linear(small_n_hidden, small_n_hidden),
#             nn.ReLU(),
#             nn.Linear(small_n_hidden, small_n_hidden),
#             nn.ReLU(),
#             nn.Linear(small_n_hidden, small_n_hidden),
#             nn.ReLU(),
#             nn.Linear(small_n_hidden, small_output_size),
#             nn.Softmax(dim=1)
#         )
#
#         self.smallDonor = nn.Sequential(
#             nn.Linear(small_input_size, small_n_hidden),
#             nn.ReLU(),
#             nn.Linear(small_n_hidden, small_n_hidden),
#             nn.ReLU(),
#             nn.Linear(small_n_hidden, small_n_hidden),
#             nn.ReLU(),
#             nn.Linear(small_n_hidden, small_n_hidden),
#             nn.ReLU(),
#             nn.Linear(small_n_hidden, small_donor_output_size),
#             nn.Softmax(dim=1)
#         )
#
#         self.embbeing_size = emb_size
#
#         self.batchsize = 0
#
#     def BidModel(self, input_iab):
#         input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.feature_num))
#         input_iab = self.embed(input_iab)
#         input_iab = input_iab.view(-1, 10 * self.embbeing_size)
#         output = self.big(input_iab)
#         return output
#
#     def forward(self, x, UEIdx):
#         self.batchsize = x.shape[0]
#         x_iabs = x[:, 0:40]
#         x_ues = x[:, 40:]
#
#         # x_iabs = torch.reshape(x_iabs, (batchsize, -1, rp.feature_num))
#         # x_iabs = self.embed(x_iabs)
#         # x_iabs = x_iabs.view(-1, 10 * self.embbeing_size)
#         # x_iabs = self.big(x_iabs)
#         x_iabs = self.BidModel(x_iabs)
#         x_ues = x_ues.view(-1, rp.feature_num * rp.maxUEperBS * rp.IAB_num)
#         y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)
#         input_feature = rp.feature_num * rp.maxUEperBS
#         for i in range(0, rp.IAB_num):
#             small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
#             small_in_UE = torch.reshape(small_in_UE, (self.batchsize, -1, rp.feature_num))
#             small_in_UE = self.embed(small_in_UE)   # Embbedind input
#             small_in_UE = torch.reshape(small_in_UE, (-1, rp.maxUEperBS * self.embbeing_size))
#             small_in_IAB = x_iabs[:, i:i + 1]   # take BW allocated to the IAB
#             small_in = torch.cat((small_in_UE, small_in_IAB), dim=1)
#             small_out_UE = self.small(small_in)     # forward in the small model
#             ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
#             # Normalize output according to UE number in IAB
#             temp = torch.sum((small_out_UE * ueOnIndicator), dim=1)
#             temp = temp.repeat((rp.maxUEperBS + rp.backhaul_num)*2, 1)
#             temp = torch.transpose(temp, 0, 1) + rp.eps
#             y[:, i, :] = ((small_out_UE * ueOnIndicator) / temp) * small_in_IAB
#         return y


class resourceAllocation_Dynamic(nn.Module):
    def __init__(self,
                 unit_input=4,
                 emb_size=20,
                 small_input_size=20 * rp.maxUEperBS + 1,
                 small_n_hidden=20 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=20 * rp.IAB_num,
                 big_n_hidden=20 * rp.IAB_num, big_output_size=10):
        super(resourceAllocation_Dynamic, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. wil change in the beginning of the forward

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:40]
        x_ues = x[:, 40:]

        x_iabs = self.big_model(x_iabs)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class resourceAllocation_Dynamic1(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=30,
                 small_input_size=30 * rp.maxUEperBS + 1,
                 small_n_hidden=30 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=30 * rp.IAB_num,
                 big_n_hidden=30 * rp.IAB_num, big_output_size=10):
        super(resourceAllocation_Dynamic1, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]


        x_iabs = self.big_model(x_iabs)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocationDynamic2(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=10,
                 small_input_size=10 * rp.maxUEperBS + 1,
                 small_n_hidden=10 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=10 * rp.IAB_num,
                 big_n_hidden=10 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocationDynamic2, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.ReLU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.ReLU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]


        x_iabs = self.big_model(x_iabs)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocationDynamicGelu(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=30,
                 small_input_size=30 * rp.maxUEperBS + 1,
                 small_n_hidden=30 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=30 * rp.IAB_num,
                 big_n_hidden=30 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocationDynamicGelu, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
            nn.ReLU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]


        x_iabs = self.big_model(x_iabs)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocationDynamicGelu2(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=40,
                 small_input_size=40 * rp.maxUEperBS + 1,
                 small_n_hidden=40 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=40 * rp.IAB_num,
                 big_n_hidden=40 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocationDynamicGelu2, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
            nn.ReLU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]


        x_iabs = self.big_model(x_iabs)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation3DNN(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=40,
                 ue_encode_size=4,
                 small_input_size=40 * rp.maxUEperBS + 1,
                 small_n_hidden=40 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=40 * rp.IAB_num,
                 big_n_hidden=40 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation3DNN, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
            nn.GELU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )

        self.encode_ue = nn.Sequential(
            nn.Linear(in_features=rp.train_feature*rp.access_num, out_features=rp.train_feature*rp.access_num),
            nn.GELU(),
            nn.Linear(in_features=rp.train_feature*rp.access_num, out_features=ue_encode_size),
        )

        self.big = nn.Sequential(
            nn.Linear(big_input_size + rp.IAB_num * ue_encode_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        self.ue_encode_size = ue_encode_size

    def big_model(self, input_iab):
        # input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        # input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * (self.embbeing_size + self.ue_encode_size))
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.small(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def add_big_feature(self, input_ue, input_iab):
        input_feature = rp.train_feature * rp.maxUEperBS

        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        y = torch.zeros(self.batchsize, rp.IAB_num, self.embbeing_size + self.ue_encode_size)
        for i in range(0, rp.IAB_num):
            small_in_UE = input_ue[:, 0 + i * input_feature:input_feature + i * input_feature]
            small_in_UE1 = torch.reshape(small_in_UE, (self.batchsize, rp.IAB_num, rp.train_feature*rp.access_num))
            small_in_UE2 = self.encode_ue(small_in_UE1)  # Embbedind input
            y[:, i, :] = torch.cat((input_iab[:, i, :], small_in_UE2[:, i, :]), dim=1)
        return y


    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]
        a = self.add_big_feature(x_ues, x_iabs)
        a = a.to(device)
        x_iabs = self.big_model(a)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation3DNN_v2(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=4,
                 ue_encode_size=4,
                 small_input_size=4 * rp.maxUEperBS + 1,
                 small_n_hidden=40 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=4 * rp.IAB_num,
                 big_n_hidden=40 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation3DNN_v2, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
            nn.GELU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )

        self.encode_ue = nn.Sequential(
            nn.Linear(in_features=rp.train_feature*rp.access_num, out_features=rp.train_feature*rp.access_num),
            nn.GELU(),
            nn.Linear(in_features=rp.train_feature*rp.access_num, out_features=ue_encode_size),
        )

        self.big = nn.Sequential(
            nn.Linear(big_input_size + rp.IAB_num * ue_encode_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        self.ue_encode_size = ue_encode_size

    def big_model(self, input_iab):
        # input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        # input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * (self.embbeing_size + self.ue_encode_size))
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.small(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def add_big_feature(self, input_ue, input_iab):
        input_feature = rp.train_feature * rp.maxUEperBS

        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        y = torch.zeros(self.batchsize, rp.IAB_num, self.embbeing_size + self.ue_encode_size)
        for i in range(0, rp.IAB_num):
            small_in_UE = input_ue[:, 0 + i * input_feature:input_feature + i * input_feature]
            small_in_UE1 = torch.reshape(small_in_UE, (self.batchsize, rp.IAB_num, rp.train_feature*rp.access_num))
            small_in_UE2 = self.encode_ue(small_in_UE1)  # Embbedind input
            y[:, i, :] = torch.cat((input_iab[:, i, :], small_in_UE2[:, i, :]), dim=1)
        return y


    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]
        a = self.add_big_feature(x_ues, x_iabs)
        a = a.to(device)
        x_iabs = self.big_model(a)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation3DNN_v3(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=4,
                 ue_encode_size=4,
                 small_input_size=4 * rp.maxUEperBS + 1,
                 small_n_hidden=40 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=4 * rp.IAB_num,
                 big_n_hidden=40 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation3DNN_v3, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
        )

        self.encode_ue = nn.Sequential(
            nn.Linear(in_features=rp.train_feature*rp.access_num, out_features=ue_encode_size),
        )

        self.big = nn.Sequential(
            nn.Linear(big_input_size + rp.IAB_num * ue_encode_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        self.ue_encode_size = ue_encode_size

    def big_model(self, input_iab):
        # input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        # input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * (self.embbeing_size + self.ue_encode_size))
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.small(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def add_big_feature(self, input_ue, input_iab):
        input_feature = rp.train_feature * rp.maxUEperBS

        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        y = torch.zeros(self.batchsize, rp.IAB_num, self.embbeing_size + self.ue_encode_size)
        for i in range(0, rp.IAB_num):
            small_in_UE = input_ue[:, 0 + i * input_feature:input_feature + i * input_feature]
            small_in_UE1 = torch.reshape(small_in_UE, (self.batchsize, rp.IAB_num, rp.train_feature*rp.access_num))
            small_in_UE2 = self.encode_ue(small_in_UE1)  # Embbedind input
            y[:, i, :] = torch.cat((input_iab[:, i, :], small_in_UE2[:, i, :]), dim=1)
        return y


    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]
        a = self.add_big_feature(x_ues, x_iabs)
        a = a.to(device)
        x_iabs = self.big_model(a)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation3DNN_v4(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=10,
                 ue_encode_size=4,
                 small_input_size=10 * rp.maxUEperBS + 1,
                 small_n_hidden=40 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=10 * rp.IAB_num,
                 big_n_hidden=40 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation3DNN_v4, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
        )

        self.encode_ue = nn.Sequential(
            nn.Linear(in_features=rp.train_feature*rp.access_num, out_features=ue_encode_size),
        )

        self.big = nn.Sequential(
            nn.Linear(big_input_size + rp.IAB_num * ue_encode_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )


        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        self.ue_encode_size = ue_encode_size

    def big_model(self, input_iab):
        # input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        # input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * (self.embbeing_size + self.ue_encode_size))
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.small(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def add_big_feature(self, input_ue, input_iab):
        input_feature = rp.train_feature * rp.maxUEperBS

        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        y = torch.zeros(self.batchsize, rp.IAB_num, self.embbeing_size + self.ue_encode_size)
        for i in range(0, rp.IAB_num):
            small_in_UE = input_ue[:, 0 + i * input_feature:input_feature + i * input_feature]
            small_in_UE1 = torch.reshape(small_in_UE, (self.batchsize, rp.IAB_num, rp.train_feature*rp.access_num))
            small_in_UE2 = self.encode_ue(small_in_UE1)  # Embbedind input
            y[:, i, :] = torch.cat((input_iab[:, i, :], small_in_UE2[:, i, :]), dim=1)
        return y


    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]
        a = self.add_big_feature(x_ues, x_iabs)
        a = a.to(device)
        x_iabs = self.big_model(a)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation3DNN_v5(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=10,
                 ue_encode_size=4,
                 small_input_size=10 * rp.maxUEperBS + 1,
                 small_n_hidden=40 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=10 * rp.IAB_num,
                 big_n_hidden=40 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation3DNN_v5, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
        )

        self.encode_ue = nn.Sequential(
            nn.Linear(in_features=rp.train_feature*rp.access_num, out_features=ue_encode_size),
        )

        self.big = nn.Sequential(
            nn.Linear(big_input_size + rp.IAB_num * ue_encode_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )


        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        self.ue_encode_size = ue_encode_size

    def big_model(self, input_iab):
        # input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        # input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * (self.embbeing_size + self.ue_encode_size))
        output = self.big(input_iab)
        return output

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.small(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def add_big_feature(self, input_ue, input_iab):
        input_feature = rp.train_feature * rp.maxUEperBS

        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        y = torch.zeros(self.batchsize, rp.IAB_num, self.embbeing_size + self.ue_encode_size)
        for i in range(0, rp.IAB_num):
            small_in_UE = input_ue[:, 0 + i * input_feature:input_feature + i * input_feature]
            small_in_UE1 = torch.reshape(small_in_UE, (self.batchsize, rp.IAB_num, rp.train_feature*rp.access_num))
            small_in_UE2 = self.encode_ue(small_in_UE1)  # Embbedind input
            y[:, i, :] = torch.cat((input_iab[:, i, :], small_in_UE2[:, i, :]), dim=1)
        return y


    def forward(self, x, UEIdx):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]
        a = self.add_big_feature(x_ues, x_iabs)
        a = a.to(device)
        x_iabs = self.big_model(a)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2*rp.backhaul_num)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation_GNN(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=30,
                 small_input_size=30 * rp.maxUEperBS + 1,
                 small_n_hidden=30 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=30 * rp.IAB_num,
                 big_n_hidden=30 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation_GNN, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
            nn.ReLU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        # GCN layesr
        self.conv1 = GCNConv(7, 1 * 16)
        self.conv2 = GCNConv(1 * 16, 1*16)
        self.fc1 = nn.Linear(1 * 16 * 10, 10)
        self.out = nn.Softmax(dim=1)

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big_gnn(input_iab)
        return output

    def big_gcn(self, data, batch_size):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.long()
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        # Layer 3
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        # Output Layer
        x = self.out(x)
        return x

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx, iab_graph):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        # x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]

        # x_iabs = self.big_model(x_iabs)
        x_iabs = self.big_gcn(iab_graph, self.batchsize)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation_GCNConv(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=30,
                 small_input_size=30 * rp.maxUEperBS + 1,
                 small_n_hidden=30 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=30 * rp.IAB_num,
                 big_n_hidden=30 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation_GCNConv, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
            nn.GELU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        # GCN layesr
        self.conv1 = GCNConv(7, 2 * 16)
        self.conv2 = GCNConv(2 * 16, 4 * 16)
        self.fc1 = nn.Linear(4 * 16 * 10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.out = nn.Softmax(dim=1)

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big_gnn(input_iab)
        return output

    def big_gcn(self, data, batch_size):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.long()
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        # Layer 3
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        # Output Layer
        x = self.out(x)
        return x

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx, iab_graph):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        # x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]

        # x_iabs = self.big_model(x_iabs)
        x_iabs = self.big_gcn(iab_graph, self.batchsize)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation_SAGEConv(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=30,
                 small_input_size=30 * rp.maxUEperBS + 1,
                 small_n_hidden=30 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=30 * rp.IAB_num,
                 big_n_hidden=30 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation_SAGEConv, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
            nn.ReLU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        # GCN layesr
        self.conv1 = SAGEConv(7, 1 * 16)
        self.conv2 = SAGEConv(1 * 16, 1*16)
        self.fc1 = nn.Linear(1 * 16 * 10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.out = nn.Softmax(dim=1)

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big_gnn(input_iab)
        return output

    def big_gcn(self, data, batch_size):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.long()
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        # Layer 3
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        # Output Layer
        x = self.out(x)
        return x

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx, iab_graph):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        # x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]

        # x_iabs = self.big_model(x_iabs)
        x_iabs = self.big_gcn(iab_graph, self.batchsize)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y


class ResourceAllocation_GATv2Conv(nn.Module):
    def __init__(self,
                 unit_input=6,
                 emb_size=30,
                 small_input_size=30 * rp.maxUEperBS + 1,
                 small_n_hidden=30 * rp.maxUEperBS + 1,
                 small_output_size=(rp.maxUEperBS + rp.backhaul_num)*2,
                 small_donor_output_size=rp.maxUEperBS*2,
                 big_input_size=30 * rp.IAB_num,
                 big_n_hidden=30 * rp.IAB_num, big_output_size=10):
        super(ResourceAllocation_GATv2Conv, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_features=unit_input, out_features=emb_size),
            nn.ReLU(),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )
        self.big = nn.Sequential(
            nn.Linear(big_input_size, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_n_hidden),
            nn.GELU(),
            nn.Linear(big_n_hidden, big_output_size),
            nn.Softmax(dim=1)
        )

        self.small = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.smallDonor = nn.Sequential(
            nn.Linear(small_input_size, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_n_hidden),
            nn.GELU(),
            nn.Linear(small_n_hidden, small_output_size),
            nn.Softmax(dim=1)
        )

        self.embbeing_size = emb_size

        self.batchsize = 0  # initial value. will change in the beginning of the forward

        # GCN layesr
        self.conv1 = GATv2Conv(7, 1*4, heads=3)
        self.conv2 = GATv2Conv(3*4, 3*4, heads=3)
        self.fc1 = nn.Linear(3*3*4*10, 10)
        self.out = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.25)

    def big_model(self, input_iab):
        input_iab = torch.reshape(input_iab, (self.batchsize, -1, rp.train_feature))
        input_iab = self.embed(input_iab)
        input_iab = input_iab.view(-1, 10 * self.embbeing_size)
        output = self.big_gnn(input_iab)
        return output

    def big_gcn(self, data, batch_size):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.long()
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)
        # Layer 3
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        # Output Layer
        x = self.out(x)
        return x

    def small_model(self, input_ue, input_iab, input_Indicator, IABInd):
        input_ue = torch.reshape(input_ue, (self.batchsize, -1, rp.train_feature))
        input_ue = self.embed(input_ue)  # Embbedind input
        input_ue = torch.reshape(input_ue, (-1, rp.maxUEperBS * self.embbeing_size))
        input_ue = torch.cat((input_ue, input_iab), dim=1)
        if IABInd == 1:
            input_ue = self.small(input_ue)  # forward in the small model
        else:
            input_ue = self.smallDonor(input_ue)  # forward in the small model
        # Normalize output according to UE number in IAB
        NormFactor = torch.sum((input_ue * input_Indicator), dim=1)
        NormFactor = NormFactor.repeat((rp.maxUEperBS + rp.backhaul_num) * 2, 1)
        NormFactor = torch.transpose(NormFactor, 0, 1) + rp.eps
        output = ((input_ue * input_Indicator) / NormFactor) * input_iab
        return output

    def forward(self, x, UEIdx, iab_graph):
        self.batchsize = x.shape[0]
        input_feature = rp.train_feature * rp.maxUEperBS
        # x_iabs = x[:, 0:60]
        x_ues = x[:, 60:]

        # x_iabs = self.big_model(x_iabs)
        x_iabs = self.big_gcn(iab_graph, self.batchsize)
        x_ues = x_ues.view(-1, rp.train_feature * rp.maxUEperBS * rp.IAB_num)
        y = torch.zeros(self.batchsize, rp.IAB_num, (rp.maxUEperBS + rp.backhaul_num)*2)

        for i in range(0, rp.IAB_num):
            if i < rp.IAB_num - 1:
                IABnode = 1
            else:
                IABnode = 1
            small_in_IAB = x_iabs[:, i:i + 1]  # take BW allocated to the IAB
            small_in_UE = x_ues[:, 0 + i * input_feature:input_feature + i * input_feature] # Take UE
            ueOnIndicator = torch.cat((UEIdx[:, i, :], torch.ones((self.batchsize, 2)).to(device)), dim=1)
            y[:, i, :] = self.small_model(small_in_UE, small_in_IAB, ueOnIndicator, IABInd=IABnode)
        return y