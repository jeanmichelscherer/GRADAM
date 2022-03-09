#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:17:30 2022

@author: scherer
"""
import subprocess
from time import ctime
import os

def get_version_date(package_name = 'gradam'):
    package_param = subprocess.check_output(
                ['pip', 'show', package_name], stderr=subprocess.STDOUT)
    # subprocess will return a string of type `bytes` with 
    # name, version, location, and we need to get Location, like so:
    
    # Name: requests
    # Version: 1.4.5
    # another params
    # Location: /my/env/lib/python2.7/site-packages
    
    # here we decode pack, because this is a bytes, and spltit by '\n'
    # go through this list of strings and check if we have `Location` in string
    # if we have location we get this location and with `oc.path.getctime()`
    # get the time when this package was created
    date = ''
    for param in package_param.decode().split('\n'):
        if 'Version' in param:
            version = param[9:]
        if 'Location' in param:
            loc = param.split(':')[1].strip()
            #print("{}: {}".format(package_name, time.ctime(os.path.getctime(loc))))
            date = ctime(os.path.getctime(loc))
    return version, date