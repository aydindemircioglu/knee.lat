#!/usr/bin/python3
from termcolor import colored
import collections
import requests, subprocess
import pandas as pd
import sys
import os
import importlib
import configparser
from glob import glob
import sys, time, os
import shutil


def recreatePath (path):
    print ("Recreating path ", path)
    try:
        shutil.rmtree (path)
    except:
        pass
    os.makedirs (path)
    pass

 #
