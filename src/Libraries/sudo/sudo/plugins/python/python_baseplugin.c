/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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

PyTypeObject *sudo_type_Plugin = NULL;

static PyObject *
_sudo_Plugin__Init(PyObject *py_self, PyObject *py_args, PyObject *py_kwargs)
{
    debug_decl(_sudo_Plugin__Init, PYTHON_DEBUG_C_CALLS);

    py_debug_python_call("Plugin", "__init__", py_args, NULL, PYTHON_DEBUG_C_CALLS);

    if (!PyArg_UnpackTuple(py_args, "sudo.Plugin.__init__", 1, 1, &py_self))
        goto cleanup;

    Py_ssize_t pos = 0;
    PyObject *py_key = NULL, *py_value = NULL; // -> borrowed references

    while (PyDict_Next(py_kwargs, &pos, &py_key, &py_value)) {
        if (PyObject_SetAttr(py_self, py_key, py_value) != 0)
            goto cleanup;
    }

cleanup:
    if (PyErr_Occurred())
        debug_return_ptr(NULL);

    debug_return_ptr_pynone;
}


static PyMethodDef _sudo_Plugin_class_methods[] = {
    {"__init__", (PyCFunction)_sudo_Plugin__Init,
                 METH_VARARGS | METH_KEYWORDS,
                 "Base sudo plugin constructor"},
    {NULL, NULL, 0, NULL}
};


int
sudo_module_register_baseplugin(PyObject *py_module)
{
    debug_decl(sudo_module_register_baseplugin, PYTHON_DEBUG_INTERNAL);
    int rc = SUDO_RC_ERROR;
    PyObject *py_class = NULL;

    py_class = sudo_module_create_class("sudo.Plugin", _sudo_Plugin_class_methods, NULL);
    if (py_class == NULL)
        goto cleanup;

    if (PyModule_AddObject(py_module, "Plugin", py_class) < 0) {
        goto cleanup;
    }

    Py_INCREF(py_class);
    rc = SUDO_RC_OK;

    Py_CLEAR(sudo_type_Plugin);
    sudo_type_Plugin = (PyTypeObject *)py_class;
    Py_INCREF(sudo_type_Plugin);

cleanup:
    Py_CLEAR(py_class);
    debug_return_int(rc);
}
