/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#include "pyobjc-api.h"
#include <stdarg.h>

#import <Foundation/Foundation.h>

static PyObject*
make_opaque_capsule(PyObject* mod __attribute__((__unused__)))
{
	return PyCapsule_New((void*)1234, "objc.__opaque__", NULL);
}

static PyObject*
make_object_capsule(PyObject* mod __attribute__((__unused__)))
{
	NSObject* object = [[[NSObject alloc] init] autorelease];
	return PyCapsule_New(object, "objc.__object__", NULL);
}


static PyMethodDef mod_methods[] = {
	{
		"opaque_capsule",
		(PyCFunction)make_opaque_capsule,
		METH_NOARGS,
		0,
	},
	{
		"object_capsule",
		(PyCFunction)make_object_capsule,
		METH_NOARGS,
		0,
	},
	{ 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"pointersupport",
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

PyObject* PyInit_pointersupport(void);

PyObject*
PyInit_pointersupport(void)

#else

#define INITERROR() return
#define INITDONE() return

void initpointersupport(void);

void
initpointersupport(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("pointersupport", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	INITDONE();
}
