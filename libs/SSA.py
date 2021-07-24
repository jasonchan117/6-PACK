import torch
from torch import nn


class SSA_Sp(nn.Module):
    def __init__(self, dim):
        super(SSA_Sp, self).__init__()

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.attend = nn.Softmax(dim=-1)

    def forward(self, x):
        # x size :(1, 3, 480, 640)

        qvk = self.to_qkv(x)  # qvk size: (1, 9, 480, 640)

        # q = k = v = (1, 3, 480, 640)
        q, k, v = qvk.chunk(3, dim=1)
        # HW
        # q = q.transpose(2, 1).contiguous()  # (1, 500, 160, 125)
        qk = q.transpose(3, 1).contiguous() # (1, 640, 480, 3)
        qk = qk.view(qk.size(0), -1, qk.size(3)) #(1, 640 x 480, 3)
        # q = q.view(q.size(0), q.size(1), -1)
        # kq = k.transpose(3, 2).contiguous()
        # kq = kq.view(k.size(0), -1, kq.size(3))
        kq = q.view(q.size(0), q.size(1), -1) # (1, 3, 640 x 480)
        ms = self.attend(torch.bmm(qk, kq))  # (1, 640 x 480, 640 x 480)

        # Channels

        k = k.view(k.size(0), k.size(1), -1) #(1, 3, -1)
        q = q.transpose(3, 1).contiguous()
        q = q.view(q.size(0), -1, q.size(3)) #(1, -1, 3)
        ma = self.attend(torch.bmm(k, q))  # (1, 3, 3)

        vq = v.transpose(3,1).contiguous()
        vq = vq.view(v.size(0), -1, vq.size(3)), #(1, 640x480, 3)
        vk = v.view(v.size(0), v.size(1), -1)

        ms = torch.bmm(ms, vq)#(1, 640x480, 3)
        ms = ms.transpose(2, 1).contiguous()#(1, 3, 640x480)
        ma = torch.bmm(ma, vk)##(1, 3, 640x480)
        ms = ms.view(x.size(0), x.size(1), x.size(2), x.size(3))
        ma = ma.view(x.size(0), x.size(1), x.size(2), x.size(3))
        return ms + ma



# class CrossAttention(nn.Module):
#     def __init__(self, dim):
#         super(CrossAttention, self).__init__()
#
#         self.attend = nn.Softmax(dim=-1)
#         self.cross_ant = nn.Conv3d(dim, dim * 2, kernel_size=(3, 1, 1), padding=(1, 0, 0))
#         self.qury = nn.Conv2d(dim, dim, kernel_size=1)  # nn.Conv3d(dim, dim , kernel_size=(3, 1, 1), padding=(1, 0, 0))
#
#     def forward(self, storage, target):  # (1, 4, 125, 500, 160)  (1, 1, 125, 500, 160)
#
#         storage = storage.transpose(4, 1).transpose(4, 2).contiguous()  # (1, 160, 4, 500, 125)
#         target = target.transpose(4, 2).contiguous()  # (1, 1, 160, 500, 125)
#
#         output = self.cross_ant(storage)
#         s_v, s_k = output.chunk(2, dim=1)  # (1, 160, 4, 500, 125)
#         target = target.squeeze(1)  # (1, 160, 500, 125)
#         t_q = self.qury(target)  # (1, 160, 500, 125)
#         t_q = t_q.unsqueeze(1)  # (1, 1, 160, 500, 125)
#         t_q = t_q.transpose(2, 1).contiguous()
#         t_q = t_q.repeat(1, 1, storage.size(2), 1, 1)  # (1, 160, 4, 500, 125)
#
#         t_q = t_q.view(t_q.size(0), t_q.size(2) * t_q.size(3), -1)  # (1, 500 x 4 , 125 x 160 )
#         s_v = s_v.view(s_v.size(0), s_v.size(2) * s_v.size(3), -1)  # (1, 500 x 4, 125 x 160)
#         s_k = s_k.view(s_k.size(0), s_k.size(3) * s_k.size(2), -1).transpose(2,
#                                                                              1).contiguous()  # (1, 160 x 125, 500 x 4 )
#         kq = self.attend(torch.bmm(t_q, s_k))  # (1, 500 x 4, 500 x 4)
#
#         final = torch.bmm(kq, s_v)  # (1, 500 x 4, 160 x 125)
#         final = final.view(final.size(0), storage.size(2), storage.size(4), storage.size(3),
#                            storage.size(1))  # (1, 4, 125, 500, 160)
#
#         return final

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()

        self.attend = nn.Softmax(dim=-1)
        self.cross_ant = nn.Conv2d(dim, dim * 2, kernel_size=1 )
        self.qury = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, storage, target):
        # (1, 4, 125, 160)
        # (1, 1, 125, 160)

        output = self.cross_ant(storage)
        s_v, s_k = output.chunk(2, dim=1)  # (1, 4, 125, 160)

        t_q = self.qury(target)  # (1, 1, 125, 160)

        t_q = t_q.repeat(1, s_v.size(1), 1, 1).contiguous() # (1, 4, 125, 160)
        s_k = s_k.view(s_k.size(0), s_k.size(1), -1) # (1, 4, -1)
        t_q = t_q.view(t_q.size(0), t_q.size(1), -1) # (1, 4, -1)
        s_k = s_k.transpose(2, 1).contiguous() # (1, -1, 4)
        kq = self.attend(torch.bmm(t_q, s_k))  # (1, 4, 4)

        s_v = s_v.view(s_v.size(0), s_v.size(1), -1) #(1, 4, -1)
        final = torch.bmm(kq, s_v)  # (1, 4, -1)
        final = final.view(final.size(0), storage.size(1), storage.size(2), storage.size(3)) #(1, 4, 125, 160)

        return final