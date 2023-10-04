import os
import sys
# import numpy as np


# Class derived from Exception for realizability issue
class RealizabilityErr(Exception):

    def __init__(self, moments, negValue=None, negNode=False):
        if negNode:
            self.message = '\nError:\nNegative node "' + str(negValue) + \
                '" is calculated for the following set of moments:\n' + \
                str(moments)

        else:
            self.message = '\nError:\nRealizability issue with the ' + \
                'following set of moments:\n' + str(moments)

    def __str__(self):
        return self.message


# Function to modify string property of unforeseen exceptions raised in runtime
def create_unhandled_exception(raisedException):

    return type(raisedException)(
        '\nUnhandled exception happened: \n' + str(raisedException) + '\n'
        ).with_traceback(sys.exc_info()[2])
