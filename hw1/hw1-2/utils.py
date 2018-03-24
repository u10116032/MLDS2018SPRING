#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:58:56 2018

@author: halley
"""

import tensorflow as tf


def safe_log(x, eps= 1e-14):
    return tf.log(x + eps)

    