#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlalchemy.types as types


# https://stackoverflow.com/questions/6262943/sqlalchemy-how-to-make-django-choices-using-sqlalchemy
class ChoiceType(types.TypeDecorator):
    impl = types.String

    def __init__(self, choices):
        self.choices = dict(choices)
        super(ChoiceType, self).__init__()

    def process_bind_params(self, value, dialect):
        return [k for k, v in self.choices.items() if v == value][0]

    def process_result_value(self, value, dialect):
        return self.choices[value]
