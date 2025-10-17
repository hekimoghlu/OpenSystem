/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#include "multipack.h"
static PyObject *odepack_error;
#include "__odepack.h"
static struct PyMethodDef odepack_module_methods[] = {
{"odeint", (PyCFunction) odepack_odeint, METH_VARARGS|METH_KEYWORDS, doc_odeint},
{NULL,		NULL, 0, NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_odepack",
    NULL,
    -1,
    odepack_module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__odepack(void)
{
    PyObject *m, *d, *s;

    m = PyModule_Create(&moduledef);
    import_array();
    d = PyModule_GetDict(m);

    s = PyUString_FromString(" 1.9 ");
    PyDict_SetItemString(d, "__version__", s);
    odepack_error = PyErr_NewException ("odpack.error", NULL, NULL);
    Py_DECREF(s);
    PyDict_SetItemString(d, "error", odepack_error);
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module odepack");
    }
    return m;
}
#else
PyMODINIT_FUNC init_odepack(void) {
  PyObject *m, *d, *s;
  m = Py_InitModule("_odepack", odepack_module_methods);
  import_array();
  d = PyModule_GetDict(m);

  s = PyUString_FromString(" 1.9 ");
  PyDict_SetItemString(d, "__version__", s);
  odepack_error = PyErr_NewException ("odepack.error", NULL, NULL);
  Py_DECREF(s);
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module odepack");
}
#endif    
