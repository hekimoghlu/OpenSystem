/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
#import <AppKit/AppKit.h>


static PyObjC_function_map function_map[] = {
#if PyObjC_BUILD_RELEASE >= 1008
	{ "NSEdgeInsetsMake", (PyObjC_Function_Pointer)&NSEdgeInsetsMake },
#endif
	{ "NSEventMaskFromType", (PyObjC_Function_Pointer)&NSEventMaskFromType },
    { 0, 0 }
};

static PyMethodDef mod_methods[] = {
        { 0, 0, 0, 0 } /* sentinel */
};


/* Python glue */
#if PY_MAJOR_VERSION == 3

static struct PyModuleDef mod_module = {
        PyModuleDef_HEAD_INIT,
	"_inlines",
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

PyObject* PyInit__inlines(void);

PyObject*
PyInit__inlines(void)

#else

#define INITERROR() return
#define INITDONE() return

void init_inlines(void);

void
init_inlines(void)
#endif
{
	PyObject* m;
#if PY_MAJOR_VERSION == 3
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("_inlines", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) { 
		INITERROR();
	}

	if (PyModule_AddObject(m, "_inline_list_", 
		PyObjC_CreateInlineTab(function_map)) < 0) {
		INITERROR();
	}

	INITDONE();
}
