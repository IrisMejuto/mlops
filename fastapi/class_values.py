from typing import Optional
from pydantic import BaseModel

class UserData(BaseModel): 
    name: str
    surname: str 
    address: str 
    phone_number: int 
    second_phone_number: Optional[str] = None
