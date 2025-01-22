from aiogram.fsm.state import State, StatesGroup

class Form(StatesGroup):
    weight = State()
    height = State()
    age = State()
    a_time = State()
    city = State()