# preprocessing/vocab_builder.py
from utils.conversion_utils import hex_to_bytes

class VocabBuilder:
    @staticmethod
    def build_vocab(messages_hex):
        """
        构建字节值（0-255）到索引的映射。
        索引1保留给<PAD>，有效字节从2开始。
        """
        byte_set = set()
        for msg in messages_hex:
            byte_data = hex_to_bytes(msg)
            for b in byte_data:
                byte_set.add(b)
        # 排序保证可复现
        sorted_bytes = sorted(byte_set)
        vocab = {b: i+2 for i, b in enumerate(sorted_bytes)}  # 索引从2开始
        vocab['<PAD>'] = 1   # 填充符索引为1
        # 注意：字节值0也会被映射到某个>=2的索引，不会与<PAD>冲突
        return vocab

    @staticmethod
    def get_pad_idx():
        return 1