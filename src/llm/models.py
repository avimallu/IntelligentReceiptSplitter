from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel

from src.llm.utils import is_valid_currency_code


class Message(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str]
    images: Optional[list[str] | bytes | list[Path]]


class ReceiptMerchant(BaseModel):
    name: str


class ReceiptDate(BaseModel):
    date: date


class ReceiptAmount(BaseModel):
    currency: Annotated[str, is_valid_currency_code]
    amount: float


class Amount(TypedDict):
    currency: Annotated[str, is_valid_currency_code]
    amount: float


class ReceiptItemized(BaseModel):
    class ReceiptLineItemAmount(BaseModel):
        name: str
        currency: Annotated[str, is_valid_currency_code]
        amount: float

    ItemizedReceipt: list[ReceiptLineItemAmount]


class ItemizedAmounts(TypedDict):
    name: str
    currency: Annotated[str, is_valid_currency_code]
    amount: float


class ReceiptExtracted(TypedDict):
    merchant: str
    receipt_date: date
    total: Amount
    tip: Amount
    tax: Amount
    receipt_items: list[ItemizedAmounts]
