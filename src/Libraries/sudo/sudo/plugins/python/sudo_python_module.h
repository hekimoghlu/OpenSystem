/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef SUDO_PYTHON_MODULE_H
#define SUDO_PYTHON_MODULE_H

#include "pyhelpers.h"

extern PyObject *sudo_exc_SudoException;  // Base exception for the sudo module problems

// This is for the python plugins to report error messages for us
extern PyObject *sudo_exc_PluginException;  // base exception of the following:
extern PyObject *sudo_exc_PluginReject;  // a reject with message
extern PyObject *sudo_exc_PluginError;   // an error with message

extern PyTypeObject *sudo_type_Plugin;
extern PyTypeObject *sudo_type_ConvMessage;

extern PyObject *sudo_type_LogHandler;

PyObject *sudo_module_create_class(const char *class_name, PyMethodDef *class_methods,
                                   PyObject *base_class);

CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION
int sudo_module_register_conv_message(PyObject *py_module);

CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION
int sudo_module_ConvMessage_to_c(PyObject *py_conv_message, struct sudo_conv_message *conv_message);

CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION
int sudo_module_ConvMessages_to_c(PyObject *py_tuple, Py_ssize_t *num_msgs, struct sudo_conv_message **msgs);

CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION
int sudo_module_register_baseplugin(PyObject *py_module);

CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION
int sudo_module_register_loghandler(PyObject *py_module);

CPYCHECKER_NEGATIVE_RESULT_SETS_EXCEPTION
int sudo_module_set_default_loghandler(void);

PyObject *python_sudo_debug(PyObject *py_self, PyObject *py_args);

PyMODINIT_FUNC sudo_module_init(void);

#endif // SUDO_PYTHON_MODULE_H
