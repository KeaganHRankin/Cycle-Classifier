#Keagan Rankin
#2020/06/07
"""Classes in python."""

#This relates to object oriented programming, where you write classes representing real world things...
#and create objects based on the classes. Classes define the general behaviour that the category of objects have.
#each object is given the class behaviour.
#creating objects from a class = "instantiation", you work with instances of a class.

#-----------------------------------------------------------------------------------------------------------------------
#CREATING AND USING A CLASS
#most things can be modelled using classes.
#start by making a class called Dog, which represents any dog.
#Dogs have info (name and age) and behaviours (sit, rollover).
# This class will tell python how to make an object representing a dog.

class Dog:  #names of classes are capitalized by convention in python.
    """A simple model of a dog."""

    def __init__(self, name, age):
        """Initialize the dog, the name and age attribute."""
        self.name = name
        self.age = age
    #A function that's part of a class is called a method. __init__ method is run whenever an instance is created
    #based on the dog class. Self is passed automatically, so we only need to provide values for name and age.
    #self.var lets that var be accessed by every method (function) in the class, and the var (name, age) is called an
    #attribute.

    def sit(self):
      """Simulate a dog sitting in response to a command."""
      print(f"{self.name} is now sitting.")
    #these two methods don't need more info to run, so are just given the self argument.

    def roll_over(self):
        """Simulate a dog rolling over in response to a command."""
        print(f"{self.name} rolled over.")

#The Class is a set of instructions on how to make an instance. The instances represent specific dogs.
#Now make an instance representing your specific dog.
my_dog = Dog('Sadie', 5)    #lowercase refers to a specific instance.

print(f"My dog's name is {my_dog.name}")
print(f"My dog's age is {my_dog.age}")

#after creating an instance, we can use . notation to call any method defined in the class Dog.
my_dog.sit()
my_dog.roll_over()

#We can create as many instances as we need using a class.
#Each instance needs a unique name.
dog_two = Dog('Lucy', 3)
print(f"your dog is named {dog_two.name} and is {dog_two.age} years old.")
dog_two.roll_over()

#-----------------------------------------------------------------------------------------------------------------------
print('\n')

#Usually with instances, you will want to modify their attributes, either directly or using a method.
#Use the car class as an example.
class Car:
    """Representing a car."""
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer_reading = 0   #attributes can be assigned default values.

    def get_name(self):
        long_name = f"{self.year} {self.make} {self.model}"
        return long_name.title()

    def read_odometer(self):    #lets get spicy.
        """Show the car's mileage."""
        print(f"The car has {self.odometer_reading} km on it.")

    def update_odometer(self, mileage):
        """see line 98,101."""
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage
        else:
            print("The odometer reading can only go up!")

    def increment_odometer(self,miles):
        """See line 103"""
        self.odometer_reading += miles


#now make an instance and use the functions.
my_car = Car('volvo','s40','2005')
print(my_car.get_name())
my_car.read_odometer()

#try we can modify the odometer attribute.
#the easiest way is to modify directly:
my_car.odometer_reading = 100233
my_car.read_odometer()

#we can also use a method in our class to allow us to modify the odometer attribute.
my_car.update_odometer(98754)
my_car.read_odometer()
#we can do more with this function, like making sure the value can only get larger. See above.

#say you wanted to increment the value by some amount. Use the .increment method we made above.
my_car.increment_odometer(150)
my_car.read_odometer()

#-----------------------------------------------------------------------------------------------------------------------
print('\n')

#INHERITANCE
#when making a new class, you can inherit the properties of another class to take its attributes and methods.
#original class -> parent class, new class -> child class.
#lets create a class for electric cars using our Car class as a parent class.

class ElectricCar(Car):
    """specific kind of car run by electricity."""

    def __init__(self, make, model, year):
        super().__init__(make, model, year) #super() lets you use methods from the parent class, in this case the init
        #usually use the same init as parent class.
        #now we use other things from the parent class.

        #we can also add new specifics to this class
        self.battery_size = 75

    def describe_battery(self):
        print(f"This car has a {self.battery_size} -kWh battery.")


tesla = ElectricCar('tesla','model s',2019)
print(tesla.get_name()) #see how we can use this method from the parent class

#use the new methods
tesla.describe_battery()

#using the same method name as in the parent class will OVERRIDE that method to the new one.

#if we find that we there is a lot of info within a class, we can turn that object into a class
#then embed it into the larger class.
class Battery:
    """Model a battery for an electric car."""

    def __init__(self, battery_size=75):
        self.batter_size = battery_size

    def describe_battery(self):
        print(f"This car has a {self.batter_size} -kWh battery.")

    #now add as many methods here without cluttering the electric car class.
    def  get_range(self):
        if self.batter_size == 75:
            range = 260
        elif self.batter_size == 100:
            range = 310

        print(f"this car can go {range} miles on a full charge.")


#add this to the electric car class.
class ElectricCar(Car):
    """specific kind of car run by electricity."""

    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery = Battery()    #inserting the battery class into the electric car class!

#test it out:
tesla = ElectricCar('tesla','model s', 2019)
print(tesla.get_name())
tesla.battery.describe_battery()

#use new methods
tesla.battery.get_range()

#Now you cna start thinking higher level about how to relate properties and model real world objects.

#-----------------------------------------------------------------------------------------------------------------------
print('\n')

#IMPORTING CLASSES
#like functions, you can store classes as modules in another file then import it into your file.
#save the Car class into another .py file.
#importing is good for keeping code clean and working at higher level thinking.
#remember, usually we import at the start but here we will just import as the notes come.
from car import Car

my_new_car = Car('audi','a4', 2019)
print(my_new_car.get_name())
my_new_car.read_odometer()

#you can store multiple classes in a single imported file.
from car import ElectricCar

my_tesla = ElectricCar('tesla','model z', 2020)
print(my_tesla.get_name())
my_tesla.describe_battery()
#the logic is hidden in another file.

#you can import as many class as you want using:
#from car import Car, ElectricCar ..... etc etc etc etc
#or just use import car to import every module.
#importing modules that relate to each other will work.
#aliases can be used like functions with import _ as _

#find your onw approach, and keep it simple!

#-----------------------------------------------------------------------------------------------------------------------
#PYTHON STANDARD LIBRARY: can import these things for use, check em out! random for example
#SEE END OF CHAPTER FOR STYLING -> CapitalsNoUnderScore, docstrings
#check out python modules of the week.
