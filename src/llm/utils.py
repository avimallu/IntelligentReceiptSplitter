def is_valid_currency_code(value: str) -> bool:
    return len(value) == 3 and value.isupper() and value.isalpha()
