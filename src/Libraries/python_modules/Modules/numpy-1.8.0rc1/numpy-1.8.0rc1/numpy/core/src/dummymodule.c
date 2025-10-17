/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
 * This is a dummy module whose purpose is to get distutils to generate the
 * configuration files before the libraries are made.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <npy_pycompat.h>

static struct PyMethodDef methods[] = {
    {NULL, NULL, 0, NULL}
};


#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "dummy",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

/* Initialization function for the module */
#if defined(NPY_PY3K)
PyMODINIT_FUNC PyInit__dummy(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}
#else
PyMODINIT_FUNC
init_dummy(void) {
    Py_InitModule("_dummy", methods);
}
#endif
