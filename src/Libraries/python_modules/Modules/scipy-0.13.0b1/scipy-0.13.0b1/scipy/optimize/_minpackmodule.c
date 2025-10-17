/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 12, 2023.
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
#include "minpack.h"
static PyObject *minpack_error;
#include "__minpack.h"
static struct PyMethodDef minpack_module_methods[] = {
{"_hybrd", minpack_hybrd, METH_VARARGS, doc_hybrd},
{"_hybrj", minpack_hybrj, METH_VARARGS, doc_hybrj},
{"_lmdif", minpack_lmdif, METH_VARARGS, doc_lmdif},
{"_lmder", minpack_lmder, METH_VARARGS, doc_lmder},
{"_chkder", minpack_chkder, METH_VARARGS, doc_chkder},
{NULL,		NULL, 0, NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_minpack",
    NULL,
    -1,
    minpack_module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
PyObject *PyInit__minpack(void)
{
    PyObject *m, *d, *s;

    m = PyModule_Create(&moduledef);
    import_array();

    d = PyModule_GetDict(m);

    s = PyUnicode_FromString(" 1.10 ");
    PyDict_SetItemString(d, "__version__", s);
    Py_DECREF(s);
    minpack_error = PyErr_NewException ("minpack.error", NULL, NULL);
    PyDict_SetItemString(d, "error", minpack_error);
    if (PyErr_Occurred())
        Py_FatalError("can't initialize module minpack");

    return m;
}
#else
PyMODINIT_FUNC init_minpack(void) {
  PyObject *m, *d, *s;
  m = Py_InitModule("_minpack", minpack_module_methods);
  import_array();
  d = PyModule_GetDict(m);

  s = PyString_FromString(" 1.10 ");
  PyDict_SetItemString(d, "__version__", s);
  Py_DECREF(s);
  minpack_error = PyErr_NewException ("minpack.error", NULL, NULL);
  PyDict_SetItemString(d, "error", minpack_error);
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module minpack");
}
#endif        
