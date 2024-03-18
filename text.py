import json
import re
import torch
from torch import Tensor
from abc import ABC, abstractmethod

#Số thực là dữ liệu 0D, vector 1D, ma trận 2D còn dữ liệu từ 3D trở đi được gọi là tensor

#khởi tạo lớp trừu tượng
class Text(ABC):
    def __init__(self):
        self.pad_id = 0 #sử dụng để thêm vào các chuỗi để đảm bảo rằng chúng có cùng độ dài.
        self.sos_id = 1 #token bắt đầu (Start of Sentence)
        self.eos_id = 2 #token kết thúc (End of Sentence)

    """
    abstractmethod dùng để đánh dấu một phương thức là trừu tượng.
    Phương thức trừu tượng không có cơ thể và phải được định nghĩa lại trong các lớp con.
    """
    @abstractmethod
    def tokenize(self, formula: str):
        pass

    #chuyển đổi một tensor số nguyên thành một chuỗi văn bản
    def int2text(self, x: Tensor):
        return " ".join([self.id2word[i] for i in x if i > self.eos_id])

    #chuyển đổi một chuỗi văn bản thành một tensor số nguyên    
    def text2int(self, formula: str):
        return torch.LongTensor([self.word2id[i] for i in self.tokenize(formula)])


class Text100k(Text):
    def __init__(self):
        super().__init__()
        self.id2word = json.load(open("data/vocab/100k_vocab.json", "r")) #đọc file json
        self.word2id = dict(zip(self.id2word, range(len(self.id2word)))) #tạo một từ điển từ id2word

        """
        formula = "\\sqrt{4} + 3 * [a-zA-Z] - 5"
        [('\\sqrt', '', '', ''), ('', '', '+', ''), ('', '', '', ''), ('', '', '', '3'), ('', '', '*', ''), ('', '', '', ''), ('', '', '', '['), ('', '', '', 'a'), ('', '', '', '-'), ('', '', '', 'Z'), ('', '', '', ']'), ('', '', '-', ''), ('', '', '', '5'), ('', '', '', '')]
        [\, [, a, -, Z, ]: Đây là một loạt các ký tự đặc biệt hoặc ký tự bắt đầu bằng \, nên chúng được nhóm lại thành các token đặc biệt.
        """

        self.TOKENIZE_PATTERN = re.compile(
            "(\\\\[a-zA-Z]+)|" + '((\\\\)*[$-/:-?{-~!"^_`\[\]])|' + "(\w)|" + "(\\\\)"
        ) #tạo một biểu thức chính quy
        self.n_class = len(self.id2word) #số lớp

    def tokenize(self, formula: str):
        tokens = re.finditer(self.TOKENIZE_PATTERN, formula) #tìm kiếm tất cả các chuỗi con khớp với biểu thức chính quy
        tokens = list(map(lambda x: x.group(0), tokens))
        tokens = [x for x in tokens if x is not None and x != ""]
        return tokens #sau đó trả về danh sách các token.


class Text170k(Text):
    def __init__(self):
        super().__init__()
        self.id2word = json.load(open("data/vocab/170k_vocab.json", "r"))
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
        self.n_class = len(self.id2word)

    def tokenize(self, formula: str):
        return formula.split() #tách chuỗi thành danh sách các chuỗi con.