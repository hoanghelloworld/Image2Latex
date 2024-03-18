import random
import torch
from torch import nn, Tensor
from .decoder import Decoder
from .encoder import *
from .text import Text


class Image2Latex(nn.Module):
    def __init__(
        self,
        n_class: int,            # Số lượng lớp đầu ra, thường là số lượng token trong từ điển của văn bản đầu ra.
        enc_dim: int = 512,      # Kích thước của không gian đặc trưng đầu ra từ bộ mã hóa.
        enc_type: str = "conv_row_encoder",  # Loại bộ mã hóa được sử dụng
        emb_dim: int = 80,       # Kích thước của embedding cho từng token đầu vào.
        dec_dim: int = 512,      # Kích thước của không gian đặc trưng đầu ra từ bộ giải mã.
        attn_dim: int = 512,     # Kích thước của không gian đặc trưng trong cơ chế attention.
        num_layers: int = 1,     # Số lượng layer của bộ giải mã.
        dropout: float = 0.1,    # Tỉ lệ dropout để áp dụng cho các layer.
        bidirectional: bool = False,  # Có sử dụng mạng nơ-ron kép chiều hay không.
        decode_type: str = "greedy",  # Phương pháp giải mã được sử dụng, mặc định là greedy search.
        text: Text = None,       # Đối tượng văn bản được sử dụng để ánh xạ giữa token và các số nguyên.
        beam_width: int = 5,     # Kích thước của dải (beam) được sử dụng trong beam search.
        sos_id: int = 1,         # ID của token bắt đầu (Start of Sentence).
        eos_id: int = 2,         # ID của token kết thúc (End of Sentence).
    ):
        #Kiểm tra loại bộ mã hóa (encoder type)
        assert enc_type in [
            "conv_row_encoder",
            "conv_encoder",
            "conv_bn_encoder",
            "resnet_encoder",
            "resnet_row_encoder",
        ], "Not found encoder"
        super().__init__()

        #Khởi tạo các bộ mã hóa (encoder) dựa trên loại bộ mã hóa được chọn
        self.n_class = n_class
        if enc_type == "conv_row_encoder":
            self.encoder = ConvWithRowEncoder(enc_dim=enc_dim)
        elif enc_type == "conv_encoder":
            self.encoder = ConvEncoder(enc_dim=enc_dim)
        elif enc_type == "conv_bn_encoder":
            self.encoder = ConvBNEncoder(enc_dim=enc_dim)
        elif enc_type == "resnet_encoder":
            self.encoder = ResNetEncoder(enc_dim=enc_dim)
        elif enc_type == "resnet_row_encoder":
            self.encoder = ResNetWithRowEncoder(enc_dim=enc_dim)
        
        #khởi tạo các thuộc tính
        enc_dim = self.encoder.enc_dim
        self.num_layers = num_layers
        self.decoder = Decoder(
            n_class=n_class,
            emb_dim=emb_dim,
            dec_dim=dec_dim,
            enc_dim=enc_dim,
            attn_dim=attn_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            sos_id=sos_id,
            eos_id=eos_id,
        )
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        assert decode_type in ["greedy", "beamsearch"]
        self.decode_type = decode_type
        self.text = text
        self.beam_width = beam_width

    #sử dụng để khởi tạo trạng thái ẩn ban đầu của bộ giải mã LSTM.
    def init_decoder_hidden_state(self, V: Tensor):
        """
            return (h, c)
        """
        encoder_mean = V.mean(dim=1)
        h = torch.tanh(self.init_h(encoder_mean))
        c = torch.tanh(self.init_c(encoder_mean))
        return h, c #trả về trạng thái ẩn ban đầu của bộ giải mã LSTM.

    #chịu trách nhiệm thực hiện quá trình lan truyền thuận của mạng nơ-ron
    def forward(self, x: Tensor, y: Tensor, y_len: Tensor):
        encoder_out = self.encoder(x) #bộ mã hóa

        hidden_state = self.init_decoder_hidden_state(encoder_out) #khởi tạo trạng thái ẩn ban đầu của bộ giải mã LSTM.

        predictions = [] #danh sách dự đoán

        #Lặp qua từng token trong chuỗi văn bản mục tiêu
        for t in range(y_len.max().item()):
            dec_input = y[:, t].unsqueeze(1)
            out, hidden_state = self.decoder(dec_input, encoder_out, hidden_state)
            predictions.append(out.squeeze(1))

        predictions = torch.stack(predictions, dim=1)
        return predictions #trả về  các dự đoán cho mỗi token trong mỗi chuỗi văn bản mục tiêu.

    #chịu trách nhiệm thực hiện quá trình giải mã chuỗi token thành chuỗi văn bản.
    def decode(self, x: Tensor, max_length: int = 150):
        predict = []
        if self.decode_type == "greedy":
            predict = self.decode_greedy(x, max_length)
        elif self.decode_type == "beamsearch":
            predict = self.decode_beam_search(x, max_length)
        return self.text.int2text(predict) #Kết quả được trả về là chuỗi ký tự LaTeX tương ứng với các dự đoán.

    #chịu trách nhiệm thực hiện quá trình giải mã chuỗi token thành chuỗi văn bản sử dụng phương pháp greedy search.
    def decode_greedy(self, x: Tensor, max_length: int = 150):
        encoder_out = self.encoder(x) #bộ mã hóa
        bs = encoder_out.size(0) #Lấy kích thước của batch(tập hợp mẫu dữ liệu) từ encoder_out

        hidden_state = self.init_decoder_hidden_state(encoder_out) #khởi tạo trạng thái ẩn ban đầu của bộ giải mã LSTM.

        y = torch.LongTensor([self.decoder.sos_id]).view(bs, -1) #token bắt đầu (Start of Sentence)

        hidden_state = self.init_decoder_hidden_state(encoder_out) #khởi tạo trạng thái ẩn ban đầu của bộ giải mã LSTM.
 
        predictions = [] #danh sách dự đoán

        #Lặp qua từng token trong chuỗi văn bản mục tiêu
        for t in range(max_length):
            out, hidden_state = self.decoder(y, encoder_out, hidden_state)

            k = out.argmax().item()

            predictions.append(k)

            y = torch.LongTensor([k]).view(bs, -1)
        return predictions #trả về dự đoán cho mỗi token trong mỗi chuỗi văn bản mục tiêu.

    #chịu trách nhiệm thực hiện quá trình giải mã chuỗi token thành chuỗi văn bản sử dụng phương pháp beam search.
    #Beam search là một thuật toán tìm kiếm trong lĩnh vực xử lý ngôn ngữ tự nhiên (NLP) và các lĩnh vực liên quan. 
    def decode_beam_search(self, x: Tensor, max_length: int = 150):
        """
            default: batch size equal to 1
        """
        encoder_out = self.encoder(x)
        bs = encoder_out.size(0)  

        hidden_state = self.init_decoder_hidden_state(encoder_out)

        list_candidate = [
            ([self.decoder.sos_id], hidden_state, 0) #sử dụng để khởi tạo danh sách các ứng viên ban đầu
        ]  

        #Lặp qua từng token trong chuỗi văn bản mục tiêu
        for t in range(max_length):
            new_candidates = []
            for inp, state, log_prob in list_candidate:
                y = torch.LongTensor([inp[-1]]).view(bs, -1).to(device=x.device)
                out, hidden_state = self.decoder(y, encoder_out, state)

                topk = out.topk(self.beam_width)
                new_log_prob = topk.values.view(-1).tolist()
                new_idx = topk.indices.view(-1).tolist()
                for val, idx in zip(new_log_prob, new_idx):
                    new_inp = inp + [idx]
                    new_candidates.append((new_inp, hidden_state, log_prob + val))

            new_candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)
            list_candidate = new_candidates[: self.beam_width]

        return list_candidate[0][0] #Trả về chuỗi token của ứng viên tốt nhất