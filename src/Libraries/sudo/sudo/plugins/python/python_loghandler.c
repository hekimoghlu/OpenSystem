/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include "sudo_python_module.h"

PyObject *sudo_type_LogHandler;


static void
_debug_plugin(int log_level, const char *log_message)
{
    debug_decl_vars(python_sudo_debug, PYTHON_DEBUG_PLUGIN);

    if (sudo_debug_needed(SUDO_DEBUG_INFO)) {
        // at trace level we output the position for the python log as well
        char *func_name = NULL, *file_name = NULL;
        long line_number = -1;

        if (py_get_current_execution_frame(&file_name, &line_number, &func_name) == SUDO_RC_OK) {
            sudo_debug_printf(SUDO_DEBUG_INFO, "%s @ %s:%ld debugs:\n",
                              func_name, file_name, line_number);
        }

        free(func_name);
        free(file_name);
    }

    sudo_debug_printf(log_level, "%s\n", log_message);
}

PyObject *
python_sudo_debug(PyObject *Py_UNUSED(py_self), PyObject *py_args)
{
    debug_decl(python_sudo_debug, PYTHON_DEBUG_C_CALLS);
    py_debug_python_call("sudo", "debug", py_args, NULL, PYTHON_DEBUG_C_CALLS);

    int log_level = SUDO_DEBUG_DEBUG;
    const char *log_message = NULL;
    if (!PyArg_ParseTuple(py_args, "is:sudo.debug", &log_level, &log_message)) {
        debug_return_ptr(NULL);
    }

    _debug_plugin(log_level, log_message);

    debug_return_ptr_pynone;
}

static int
_sudo_log_level_from_python(long level)
{
    if (level >= 50)
        return SUDO_DEBUG_CRIT;
    if (level >= 40)
        return SUDO_DEBUG_ERROR;
    if (level >= 30)
        return SUDO_DEBUG_WARN;
    if (level >= 20)
        return SUDO_DEBUG_INFO;

    return SUDO_DEBUG_TRACE;
}

static PyObject *
_sudo_LogHandler__emit(PyObject *py_self, PyObject *py_args)
{
    debug_decl(_sudo_LogHandler__emit, PYTHON_DEBUG_C_CALLS);

    PyObject *py_record = NULL; // borrowed
    PyObject *py_message = NULL;

    py_debug_python_call("LogHandler", "emit", py_args, NULL, PYTHON_DEBUG_C_CALLS);

    if (!PyArg_UnpackTuple(py_args, "sudo.LogHandler.emit", 2, 2, &py_self, &py_record))
        goto cleanup;

    long python_loglevel = py_object_get_optional_attr_number(py_record, "levelno");
    if (PyErr_Occurred()) {
        PyErr_Format(sudo_exc_SudoException, "sudo.LogHandler: Failed to determine log level");
        goto cleanup;
    }

    int sudo_loglevel = _sudo_log_level_from_python(python_loglevel);

    py_message = PyObject_CallMethod(py_self, "format", "O", py_record);
    if (py_message == NULL)
        goto cleanup;

    _debug_plugin(sudo_loglevel, PyUnicode_AsUTF8(py_message));

cleanup:
    Py_CLEAR(py_message);
    if (PyErr_Occurred()) {
        debug_return_ptr(NULL);
    }

    debug_return_ptr_pynone;
}

/* The sudo.LogHandler class can be used to make the default python logger
 * use sudo's built in log system. */
static PyMethodDef _sudo_LogHandler_class_methods[] =
{
    {"emit", _sudo_LogHandler__emit, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// This function registers sudo.LogHandler class
int
sudo_module_register_loghandler(PyObject *py_module)
{
    debug_decl(sudo_module_register_loghandler, PYTHON_DEBUG_INTERNAL);

    PyObject *py_logging_module = NULL, *py_streamhandler = NULL;

    py_logging_module = PyImport_ImportModule("logging");
    if (py_logging_module == NULL)
        goto cleanup;

    py_streamhandler = PyObject_GetAttrString(py_logging_module, "StreamHandler");
    if (py_streamhandler == NULL)
        goto cleanup;

    sudo_type_LogHandler = sudo_module_create_class("sudo.LogHandler",
        _sudo_LogHandler_class_methods, py_streamhandler);
    if (sudo_type_LogHandler == NULL)
        goto cleanup;

    if (PyModule_AddObject(py_module, "LogHandler", sudo_type_LogHandler) < 0)
        goto cleanup;

    Py_INCREF(sudo_type_LogHandler);

cleanup:
    Py_CLEAR(py_streamhandler);
    Py_CLEAR(py_logging_module);
    debug_return_int(PyErr_Occurred() ? SUDO_RC_ERROR : SUDO_RC_OK);
}

// This sets sudo.LogHandler as the default log handler:
//   logging.getLogger().addHandler(sudo.LogHandler())
int
sudo_module_set_default_loghandler(void)
{
    debug_decl(sudo_module_set_default_loghandler, PYTHON_DEBUG_INTERNAL);

    PyObject *py_loghandler = NULL, *py_logging_module = NULL,
             *py_logger = NULL, *py_result = NULL;

    py_loghandler = PyObject_CallObject(sudo_type_LogHandler, NULL);
    if (py_loghandler == NULL)
        goto cleanup;

    py_logging_module = PyImport_ImportModule("logging");
    if (py_logging_module == NULL)
        goto cleanup;

    py_logger = PyObject_CallMethod(py_logging_module, "getLogger", NULL);
    if (py_logger == NULL)
        goto cleanup;

    py_result = PyObject_CallMethod(py_logger, "addHandler", "O", py_loghandler);

cleanup:
    Py_CLEAR(py_result);
    Py_CLEAR(py_logger);
    Py_CLEAR(py_logging_module);
    Py_CLEAR(py_loghandler);

    debug_return_int(PyErr_Occurred() ? SUDO_RC_ERROR : SUDO_RC_OK);
}
