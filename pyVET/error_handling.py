# -*- coding: utf-8 -*-

#
# Licensed under the BSD-3-Clause license
# Copyright (c) 2018, Andres A. Perez Hortal

"""
Error handling module.

This module includes customized pyVET exceptions.
"""

# For python 3 portability
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import super


class GeneralException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message
    
    
        
class FileNotFoundException(GeneralException):
    """ Exception when a file is not found"""
    file_path = None
    message = None

    def __init__(self, filePath):
        super().__init__(filePath)
        self.file_path = filePath
        self.message = 'Parameter file not found:  ' + self.file_path + '\n\n'
    


class FatalError(GeneralException):
    """ Fatal error exception """

    def __init__(self, main_error_message, *details_messages):
        """ Constructor """

        super().__init__(main_error_message,
                         *details_messages)

        self.main_error_message = main_error_message

        self.message = " " + main_error_message + '\n\n'

        if len(details_messages) > 0:
            self.message += "Details:" + "\n"
        for message in details_messages:
            self.message += message + '\n'

        self.message += '\n'

    