import torch
from torch import nn, Tensor


class Attention(nn.Module): # lớp attention kế thừa từ nn.Moudule trong Pytorch
    def __init__(self, enc_dim: int = 512, dec_dim: int = 512, attn_dim: int = 512):
        super().__init__()
        self.dec_attn = nn.Linear(dec_dim, attn_dim, bias=False) # dùng hàm linear để nhân ma trận đầu vào dec_dim chiều thu được ma trận đầu ra attn_dim chiều mà không cộng thêm bias
        self.enc_attn = nn.Linear(enc_dim, attn_dim, bias=False) # tương tự hàm trên
        self.full_attn = nn.Linear(attn_dim, 1, bias=False) # này cũng vậy để cuối cùng đưa về ma trận 1 chiều 
        self.softmax = nn.Softmax(dim=-1) # dùng hàm softmax để chuẩn hóa (tức là chuẩn hóa một vector nào đó có giá trị thực về 0-1)

    def forward(self, h: Tensor, V: Tensor):# hàm linear hay softmax đều lưu và tính toán ma trận dưới dạng tensor nên đầu vào phải là tensor
        """
            input:
                h: (b, dec_dim) hidden state vector of decoder
                V: (b, w * h, enc_dim) encoder matrix representation
            output:
                context: (b, enc_dim)
        """

        attn_1 = self.dec_attn(h)
        attn_2 = self.enc_attn(V)
        attn = self.full_attn(torch.tanh(attn_1.unsqueeze(1) + attn_2)).squeeze(2)
        alpha = self.softmax(attn)
        # thực hiện các phép toán đã giải thích ở trên với h và v (decoder và encoder)
        context = (alpha.unsqueeze(2) * V).sum(dim=1) # tính ra vector contex bằng cách trọng số hóa V và alpha sau đó tính tổng
        return context
