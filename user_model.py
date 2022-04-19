from typing import Optional

from pydantic import BaseModel


class User(BaseModel):
    age: Optional[int] = None
    job: Optional[str] = None
    marital: Optional[str] = None
    education: Optional[str] = None
    default: Optional[str] = None
    balance: Optional[int] = None
    housing: Optional[str] = None
    loan: Optional[str] = None
    contact: Optional[str] = None
    day: Optional[int] = None
    month: Optional[str] = None
    duration: Optional[int] = None
    campaign: Optional[int] = None
    pdays: Optional[int] = None
    previous: Optional[int] = None
    poutcome: Optional[str] = None

