# -*- coding: utf-8 -*-

#
# Licensed under the BSD-3-Clause license
# Copyright (c) 2018, Andres A. Perez Hortal

"""
Error handling module
"""
# For python 3 portability
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

__author__ = "Andres Perez Hortal"
__copyright__ = "Copyright (c) 2017, Andres A. Perez Hortal, McGill University"
__license__ = "BSD-3-Clause License, see LICENCE.txt for more details"
__email__ = "andresperezcba@gmail.com"


class FileNotFoundException(Exception):
    """ Exception when a file is not found"""
    file_path = None
    message = None

    def __init__(self, filePath):
        Exception.__init__(self, filePath)
        self.file_path = filePath
        self.message = 'Parameter file not found:  ' + self.file_path + '\n\n'


class FatalError(Exception):
    """ Fatal error exception """

    def __init__(self, main_error_message, *details_messages):
        """ Constructor """

        super(
            FatalError,
            self).__init__(
            self,
            main_error_message,
            *details_messages)

        self.main_error_message = main_error_message

        self.message = " " + main_error_message + '\n\n'

        if len(details_messages) > 0:
            self.message += "Details:" + "\n"
        for message in details_messages:
            self.message += message + '\n'

        self.message += '\n'

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message
