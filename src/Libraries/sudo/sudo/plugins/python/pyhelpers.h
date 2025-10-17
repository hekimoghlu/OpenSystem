/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#ifndef SUDO_PLUGIN_PYHELPERS_H
#define	SUDO_PLUGIN_PYHELPERS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <config.h>
#include "sudo_compat.h"
#include "sudo_plugin.h"

#include "pyhelpers_cpychecker.h"

#include "sudo_python_debug.h"

enum SudoPluginFunctionReturnCode {
    SUDO_RC_OK = 1,
    SUDO_RC_ACCEPT = 1,
    SUDO_RC_REJECT = 0,
    SUDO_RC_ERROR = -1,
    SUDO_RC_USAGE_ERROR = -2,
};

#define INTERPRETER_MAX 32

struct PythonContext
{
    sudo_printf_t sudo_log;
    sudo_conv_t sudo_conv;
    PyThreadState *py_main_interpreter;
    size_t interpreter_count;
    PyThreadState *py_subinterpreters[INTERPRETER_MAX];
};

extern struct PythonContext py_ctx;

#define Py_TYPENAME(object) (object ? Py_TYPE(object)->tp_name : "NULL")

#define py_sudo_log(...) py_ctx.sudo_log(__VA_ARGS__)

int py_sudo_conv(int num_msgs, const struct sudo_conv_message msgs[],
                 struct sudo_conv_reply replies[], struct sudo_conv_callback *callback);

void py_log_last_error(const char *context_message);

char *py_create_string_rep(PyObject *py_object);

char *py_join_str_list(PyObject *py_str_list, const char *separator);

struct key_value_str_int
{
    const char *key;
    int value;
};

PyObject *py_dict_create_string_int(size_t count, struct key_value_str_int *key_values);

PyObject *py_from_passwd(const struct passwd *pwd);

PyObject *py_str_array_to_tuple_with_count(Py_ssize_t count, char * const strings[]);
PyObject *py_str_array_to_tuple(char * const strings[]);
char **py_str_array_from_tuple(PyObject *py_tuple);

CPYCHECKER_RETURNS_BORROWED_REF
PyObject *py_tuple_get(PyObject *py_tuple, Py_ssize_t index, PyTypeObject *expected_type);

PyObject *py_object_get_optional_attr(PyObject *py_object, const char *attr, PyObject *py_default);
long long py_object_get_optional_attr_number(PyObject *py_object, const char *attr_name);
const char *py_object_get_optional_attr_string(PyObject *py_object, const char *attr_name);

void py_object_set_attr_number(PyObject *py_object, const char *attr_name, long long number);
void py_object_set_attr_string(PyObject *py_object, const char *attr_name, const char *value);

PyObject *py_create_version(unsigned int version);

void py_debug_python_call(const char *class_name, const char *function_name,
                          PyObject *py_args, PyObject *py_kwargs, int subsystem_id);
void py_debug_python_result(const char *class_name, const char *function_name,
                            PyObject *py_args, int subsystem_id);

void str_array_free(char ***array);

int py_get_current_execution_frame(char **file_name, long *line_number, char **function_name);

void py_ctx_reset(void);

#endif // SUDO_PLUGIN_PYHELPERS_H
