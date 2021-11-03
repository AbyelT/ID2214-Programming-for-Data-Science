#numbers
v = 3.6
i = 34
c = 3.14e2
z = 2 + 3j

#boolean
b = isinstance(i, float) # checks instance

#strings
s = 'hello'
b = isinstance("i", str)

#casting
i = int(3.14)
f = float(3)
s = str(3.44)
f = float(s)
b = bool("hello")
n = bool("")
m = bool(0)

#operators: built in functionality
V = 2.0 + 2**3
x = 3
x += 2
b = 2.0 == 2 

b = (1+1 == 2 and not 4>5)
b = 4 is not 4.0
a = 3 is 1+2

#order of operations
# = 
# if-else
# or
# and
# not
# <, > ==, != etc.
#+, -

#lists: indexed, ordered and changeable
lang = ["py", "e", "sand"]
allbutfirst = lang[1:]
allbutlast = lang[:-1]
lang[1] = "R"
lang.count("r")

#tuples: indexed, ordered but not changeable
fix = ("1", "2", "3")
f = fix[1]
fixed[0] = "d"  # GIVES ERROR
"3" in fix

#sets: not index, unordered, no duplicates
s = {"a","b","c","d"}
s = s.union(set(lang)) # you can get the set of lists 
# similar operations with difference and intersection

#dictionaries: indexed, unordered and changeable
d = {"python":1, "R":2, "Jule":3}
Y = d["R"]
d["S"] = 4
list(d.keys())
list(d.values())
t = d.get("T", "default_value")

#if statemetns
if a == b:
    a = 1
elif a == c:
    a = 2
else: 
    a = 3

#for loop
for i in range(10):
    print(i)
# iterate in lists, strings, enumerates
# can also use break and continue

#while loops are the same as in java
i= 1
while i > 4:
    print(i)
    i += 1

# list comprehensions lets you create lists without for/while loops, by having foor loops INSIDE a list!
nl = [la.lower() for la in lang]

# functions, each function can have a number of arguments and return a value, end is marked with indentation
## primitive values are not affected by updates from functions, composite datatype are affected though 
## fucntions can have default arguments, you can also choose which arguments to give or which argument shild be overwritten
def test_fcuntion():
    print("hello world, again")
def diff(a=10, b=20):
    return a-b
d0 = diff()
d1 = diff(b=5)

# lambda functions: anonymos functions with one expresseion
r = (lambda x: x+1)(5)
def dervi(f,x,h):
    return f(x+h) - f(x)/h

#classes and obejcts
## classes have methods: functions that are parts of a class
## the declaration self is only used in the initialization of classes
## methodsand variables in the class are reached with dot notation
## SIMILAR TO JAVA
class DSlist:
    def __init__(self, name, year):
        self.name = name
        self.year = year
l1 = DSlist("Python", 1)
l2 = DSlist("Julia", 2)
print(l1.name)

## has special methods 
# def __Str__
# def __eq__
# def __len__
## classes can be childern of parents

#modules: files which contain python code
## import modules
import my_def as md
from my_def import DSlist
test = md.function()

#input/output
print("N:{} Y :{}".format("R, 12"))
#######   NOTES   ######
## values can be overvritten, you dont need to specify which type variables are
## the order of operations are evaluated from a specific order
## lists are dynamic and can be changed anywhere, tuples are static and inmutable
## you can cast the compsoite data strucutres such as Tuple -> Set, Dictionaries -> Lists