/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#include "sharedlib.h"

#define _STR(x) #x
#define STR(x) _STR(x)

static PyObject*
mod_function(PyObject* mod __attribute__((__unused__)), PyObject* arg)
{
	int value = PyLong_AsLong(arg);
	if (PyErr_Occurred()) {
		return NULL;
	}
	value = FUNC_NAME(value);
	return PyLong_FromLong(value);
}

static PyMethodDef mod_methods[] = {
	{
		STR(NAME),
		(PyCFunction)mod_function,
		METH_O,
		0
	},
	{ 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	STR(NAME),
	NULL,
	0,
	mod_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

#define INITERROR() return NULL
#define INITDONE() return m

#define INITFUNC PyInt_ ## NAME

PyObject* INITFUNC(void);

PyObject*
INITFUNC(void)

#else

#define INITERROR() return
#define INITDONE() return


void INITFUNC(void);

void
INITFUNC(void)
#endif

{
	PyObject* m;


#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4(STR(NAME), mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	INITDONE();
}
