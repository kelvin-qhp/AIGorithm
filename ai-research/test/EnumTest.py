from enum import Enum, unique

@unique
class Weekday(Enum):
    Sun = 0 # Sun的value被设定为0
    Mon = 1
    Tue = 2
    Wed = 3
    Thu = 4
    Fri = 5
    Sat = 6

class Gender(Enum):
    Male = 0
    Female = 1

class Student(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender

class Student2(object):
    @property
    def birth(self):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return 2015 - self._birth

if __name__ == '__main__':

    print(Weekday.Sun)
    print(Weekday.Sun.value)
    print(Weekday(5))

    bart = Student('Bart', Gender.Male)
    print(f'**{bart.name}:{bart.gender.value}')
    s = Student2()
    s.birth=100
    s.birth.getter