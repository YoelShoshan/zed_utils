#def echo(**kwargs):
#    for name, value in kwargs.items():
#        print(name, value)
#a = 15
#
#echo(a=a)
#
#b = 20
#
#echo(a+b)


import inspect
import re

import warnings

warnings.warn("using debug tricks - messing with frames - should be removed from production", UserWarning)

def echo(arg):
    frame = inspect.currentframe()
    try:
        context = inspect.getframeinfo(frame.f_back).code_context
        caller_lines = ''.join([line.strip() for line in context])
        m = re.search(r'echo\s*\((.+?)\)$', caller_lines)
        if m:
            caller_lines = m.group(1)
        print(caller_lines, arg)
    finally:
        del frame

#a = 15
#b = 20
#echo(a)
#echo(b)
#echo(a+b)