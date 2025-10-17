/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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

static PyObject*
mod_CFSetGetValues(
	PyObject* self __attribute__((__unused__)),
	PyObject* args)
{
	PyObject* pySet;
	PyObject* pyValues;
	CFSetRef set;
	void* values;
	CFIndex count;

	if (!PyArg_ParseTuple(args, "OO", &pySet, &pyValues)) {

		return NULL;
	}

	if (PyObjC_PythonToObjC(@encode(CFSetRef), pySet, &set) < 0) {
		return NULL;
	}

	if (pyValues == PyObjC_NULL) {
		values = NULL;
		count = 0;
	} else if (pyValues == Py_None){
		count = CFSetGetCount(set);
		values = malloc(sizeof(void*) * count);
		if (values == NULL) {
			PyErr_NoMemory();
			return NULL;
		}
	} else {
		PyErr_SetString(PyExc_ValueError, "values must be None of NULL");
		return NULL;
	}


	PyObjC_DURING
		CFSetGetValues( set, values);

	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);

	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) {
		if (values != NULL) {
			free(values);
		}
		return NULL;
	}

	if (values != NULL) {
		pyValues = PyObjC_CArrayToPython(@encode(id), values, count);
		free(values);
	} else {
		pyValues = Py_None;
		Py_INCREF(pyValues);
	}

	return pyValues;
}

#define COREFOUNDATION_SET_METHODS \
        { 	\
		"CFSetGetValues", 	\
		(PyCFunction)mod_CFSetGetValues, 	\
		METH_VARARGS, 	\
		NULL 	\
	},
