/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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

#include "Python.h"
#include <windows.h>

static PyObject *
_GetForegroundWindow(PyObject *module, PyObject *args)
{
    HWND handle = GetForegroundWindow();
    if (!PyArg_ParseTuple(args, ":GetForegroundWindow"))
    {
        return NULL;
    }
    return PyLong_FromSize_t((size_t)handle);
}

static PyObject *
_SetForegroundWindow(PyObject *module, PyObject *args)
{
    HWND handle;
    if (!PyArg_ParseTuple(args, "n:SetForegroundWindow", &handle))
    {
        return NULL;
    }
    if (!SetForegroundWindow(handle))
    {
        return PyErr_Format(PyExc_RuntimeError,
                            "Error setting window");
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef _windowing_methods[] =
{
    {"GetForegroundWindow", _GetForegroundWindow, METH_VARARGS},
    {"SetForegroundWindow", _SetForegroundWindow, METH_VARARGS},
    {NULL, NULL}
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_windowing",
        "",
        -1,
        _windowing_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit__windowing(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    return module;
}

#else
PyMODINIT_FUNC init_windowing()
{
    Py_InitModule("_windowing", _windowing_methods);
}
#endif
