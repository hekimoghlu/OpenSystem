/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#include <Python.h>
#include "pyobjc-api.h"

#import <AppKit/AppKit.h>

static PyObject* 
objc_NSApplicationMain(
	PyObject* self __attribute__((__unused__)), 
	PyObject* args, 
	PyObject* kwds)
{
static	char* keywords[] = { "argv", NULL };
	char** argv = NULL;
	int    argc;
	PyObject* arglist;
	int       i;
	PyObject* v;
	int       res;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:NSApplicationMain",
			keywords, &arglist)) {
		return NULL;
	}

	if (!PySequence_Check(arglist)) {
		PyErr_SetString(PyExc_TypeError, 
			"NSApplicationMain: need list of strings as argument");
		return NULL;
	}

	argc = PySequence_Size(arglist);
	argv = calloc((argc + 1), sizeof(char**));
	if (argv == NULL) {
		PyErr_SetString(PyExc_MemoryError,
			"Out of memory");
		return NULL;
	}

	for  (i = 0; i < argc; i++) {
		v = PySequence_GetItem(arglist, i);
		if (v == NULL) {
			goto error_cleanup;
		}
		if (PyUnicode_Check(v)) {
			PyObject* bytes = PyUnicode_AsEncodedString(v, 
					NULL, NULL);
			if (!bytes) {
				goto error_cleanup;
			}
			argv[i] = strdup(PyBytes_AsString(bytes));
#if PY_MAJOR_VERSION == 2
		} else if (PyString_Check(v)) {
			argv[i] = strdup(PyString_AsString(v));
#endif
		} else {
			PyErr_SetString(PyExc_TypeError, 
				"NSApplicationMain: need list of strings "
				"as argument");
			goto error_cleanup;
		}

		if (argv[i] == NULL) {
			PyErr_SetString(PyExc_MemoryError,
				"Out of memory");
			goto error_cleanup;
		}
	}

	argv[argc] = NULL;

	PyObjC_DURING
		res = NSApplicationMain(argc, (const char**)argv);
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		res = -1;
	PyObjC_ENDHANDLER

	for (i = 0; i < argc; i++) {
		free(argv[i]);
	}
	free(argv);

	if (res == -1 && PyErr_Occurred())
		return NULL;
	return PyInt_FromLong(res);

error_cleanup:
	if (argv != NULL) {
		for (i = 0; i < argc; i++) {
			if (argv[i] != NULL) {
				free(argv[i]);
				argv[i] = NULL;
			}
		}
		free(argv);
		argv = NULL;
	}

	return NULL;
}


#define APPKIT_APPMAIN_METHODS \
	{  							\
		"NSApplicationMain", 				\
		(PyCFunction)objc_NSApplicationMain, 		\
		METH_VARARGS|METH_KEYWORDS, 			\
		"int NSApplicationMain(int argc, const char *argv[]);" \
	},
