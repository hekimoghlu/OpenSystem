/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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
#include "quadpack.h"
#include "__quadpack.h"
static struct PyMethodDef quadpack_module_methods[] = {
{"_qagse", quadpack_qagse, METH_VARARGS, doc_qagse},
{"_qagie", quadpack_qagie, METH_VARARGS, doc_qagie},
{"_qagpe", quadpack_qagpe, METH_VARARGS, doc_qagpe},
{"_qawoe", quadpack_qawoe, METH_VARARGS, doc_qawoe},
{"_qawfe", quadpack_qawfe, METH_VARARGS, doc_qawfe},
{"_qawse", quadpack_qawse, METH_VARARGS, doc_qawse},
{"_qawce", quadpack_qawce, METH_VARARGS, doc_qawce},
{NULL,		NULL, 0, NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_quadpack",
    NULL,
    -1,
    quadpack_module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__quadpack(void)
{
    PyObject *m, *d, *s;

    m = PyModule_Create(&moduledef);
    import_array();
    d = PyModule_GetDict(m);

    s = PyUString_FromString(" 1.13 ");
    PyDict_SetItemString(d, "__version__", s);
    quadpack_error = PyErr_NewException ("quadpack.error", NULL, NULL);
    Py_DECREF(s);
    PyDict_SetItemString(d, "error", quadpack_error);
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module quadpack");
    }
    return m;
}
#else
PyMODINIT_FUNC init_quadpack(void) {
  PyObject *m, *d, *s;
  m = Py_InitModule("_quadpack", quadpack_module_methods);
  import_array();
  d = PyModule_GetDict(m);

  s = PyUString_FromString(" 1.13 ");
  PyDict_SetItemString(d, "__version__", s);
  quadpack_error = PyErr_NewException ("quadpack.error", NULL, NULL);
  Py_DECREF(s);
  PyDict_SetItemString(d, "error", quadpack_error);
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module quadpack");
}
#endif
