# preprocessing/cleaner.py
from config import MIN_MSG_LEN, MAX_MSG_LEN, REMOVE_DUPLICATES

class Cleaner:
    @staticmethod
    def filter_by_length(messages_hex):
        filtered = []
        for msg in messages_hex:
            byte_len = len(msg) // 2
            if MIN_MSG_LEN <= byte_len <= MAX_MSG_LEN:
                filtered.append(msg)
        return filtered

    @staticmethod
    def remove_duplicates(messages_hex):
        seen = set()
        unique = []
        for msg in messages_hex:
            if msg not in seen:
                seen.add(msg)
                unique.append(msg)
        return unique

    @staticmethod
    def clean(messages_hex):
        cleaned = Cleaner.filter_by_length(messages_hex)
        if REMOVE_DUPLICATES:
            cleaned = Cleaner.remove_duplicates(cleaned)
        print(f"Cleaned: {len(messages_hex)} -> {len(cleaned)} messages")
        return cleaned