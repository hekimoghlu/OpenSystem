/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
mod_CFCalendarAddComponents(
	PyObject* self __attribute__((__unused__)),
	PyObject* args)
{
	CFCalendarRef calendar;
	CFAbsoluteTime at;
	CFOptionFlags flags;
	char* componentDesc;
	int params[10];
	Boolean result;
	int r;

	if (PyTuple_Size(args) < 4) {
		PyErr_Format(PyExc_TypeError, 
			"Expecting at least 4 arguments, got %" 
			PY_FORMAT_SIZE_T "d", PyTuple_Size(args));
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFCalendarRef), 
		PyTuple_GetItem(args, 0), &calendar);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFAbsoluteTime), 
		PyTuple_GetItem(args, 1), &at);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFOptionFlags), 
		PyTuple_GetItem(args, 2), &flags);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(char*), 
		PyTuple_GetItem(args, 3), &componentDesc);
	if (r == -1) {
		return NULL;
	}

	if (PyTuple_Size(args) != 4 + strlen(componentDesc)) {
		PyErr_Format(PyExc_TypeError, 
			"Expecting %" PY_FORMAT_SIZE_T "d arguments, got %"
			PY_FORMAT_SIZE_T "d", 4 + strlen(componentDesc),
			PyTuple_Size(args));
		return NULL;
	}
	if (PyTuple_Size(args) > 4 + 10) {
		PyErr_SetString(PyExc_TypeError,
			"At most 10 characters supported in componentDesc");
		return NULL;
	}

	Py_ssize_t i, len;

	len = strlen(componentDesc);
	for (i = 0; i < len; i++) {
		r = PyObjC_PythonToObjC(@encode(int), 
			PyTuple_GetItem(args, 4 + i), params + i);
		if (r == -1) {
			return NULL;
		}
	}

	result = FALSE;
	PyObjC_DURING
		result = CFCalendarAddComponents(
			calendar, &at, flags, componentDesc,
			params[0], params[1], params[2], params[3],
			params[4], params[5], params[6], params[7],
			params[8], params[9]);
	
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);

	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) {
		return NULL;
	}

	PyObject* b = PyBool_FromLong(result);
	if (b  == NULL) {
		return NULL;
	}
	PyObject* a = PyObjC_ObjCToPython(@encode(CFAbsoluteTime), &at);
	if (a == NULL) {
		Py_DECREF(b);
		return NULL;
	}

	return Py_BuildValue("NN", b, a);
}

		
static PyObject*
mod_CFCalendarComposeAbsoluteTime(
	PyObject* self __attribute__((__unused__)),
	PyObject* args)
{
	CFCalendarRef calendar;
	CFAbsoluteTime at;
	char* componentDesc;
	int params[10];
	Boolean result;
	int r;

	if (PyTuple_Size(args) < 3) {
		PyErr_Format(PyExc_TypeError, 
			"Expecting at least 3 arguments, got %" 
			PY_FORMAT_SIZE_T "d", PyTuple_Size(args));
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFCalendarRef), 
		PyTuple_GetItem(args, 0), &calendar);
	if (r == -1) {
		return NULL;
	}

	if (PyTuple_GetItem(args, 1) != Py_None) {
		PyErr_SetString(PyExc_TypeError, "placeholder for 'at' must be None");
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(char*), 
		PyTuple_GetItem(args, 2), &componentDesc);
	if (r == -1) {
		return NULL;
	}

	if (PyTuple_Size(args) != 3 + strlen(componentDesc)) {
		PyErr_Format(PyExc_TypeError, 
			"Expecting %" PY_FORMAT_SIZE_T "d arguments, got %"
			PY_FORMAT_SIZE_T "d", 3 + strlen(componentDesc),
			PyTuple_Size(args));
		return NULL;
	}
	if (PyTuple_Size(args) > 3 + 10) {
		PyErr_SetString(PyExc_TypeError,
			"At most 10 characters supported in componentDesc");
		return NULL;
	}

	Py_ssize_t i, len;

	len = strlen(componentDesc);
	for (i = 0; i < len; i++) {
		r = PyObjC_PythonToObjC(@encode(int), 
			PyTuple_GetItem(args, 3 + i), params + i);
		if (r == -1) {
			return NULL;
		}
	}

	result = FALSE;
	PyObjC_DURING
		result = CFCalendarComposeAbsoluteTime(
			calendar, &at, componentDesc,
			params[0], params[1], params[2], params[3],
			params[4], params[5], params[6], params[7],
			params[8], params[9]);
	
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);

	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) {
		return NULL;
	}

	PyObject* b = PyBool_FromLong(result);
	if (b  == NULL) {
		return NULL;
	}
	PyObject* a = PyObjC_ObjCToPython(@encode(CFAbsoluteTime), &at);
	if (a == NULL) {
		Py_DECREF(b);
		return NULL;
	}

	return Py_BuildValue("NN", b, a);
}

static PyObject*
mod_CFCalendarDecomposeAbsoluteTime(
	PyObject* self __attribute__((__unused__)),
	PyObject* args)
{
	CFCalendarRef calendar;
	CFAbsoluteTime at;
	char* componentDesc;
	int params[10];
	Boolean result;
	int r;

	if (PyTuple_Size(args) < 3) {
		PyErr_Format(PyExc_TypeError, 
			"Expecting at least 3 arguments, got %" 
			PY_FORMAT_SIZE_T "d", PyTuple_Size(args));
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFCalendarRef), 
		PyTuple_GetItem(args, 0), &calendar);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFAbsoluteTime), 
		PyTuple_GetItem(args, 1), &at);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(char*), 
		PyTuple_GetItem(args, 2), &componentDesc);
	if (r == -1) {
		return NULL;
	}

	if (strlen(componentDesc) > 10) {
		PyErr_SetString(PyExc_TypeError,
			"At most 10 characters supported in componentDesc");
		return NULL;
	}

	if (PyTuple_Size(args) != 3) {
		if (PyTuple_Size(args) != 3 + strlen(componentDesc)) {
			PyErr_Format(PyExc_TypeError, 
				"Expecting %" PY_FORMAT_SIZE_T "d arguments, got %"
				PY_FORMAT_SIZE_T "d", 3 + strlen(componentDesc),
				PyTuple_Size(args));
			return NULL;
		}

		Py_ssize_t i, len;

		len = strlen(componentDesc);
		for (i = 0; i < len; i++) {
			if (PyTuple_GetItem(args, 3 + i) != Py_None) {
				PyErr_SetString(PyExc_ValueError,
					"Bad placeholder value");
				return NULL;
			}
		}
	}

	result = FALSE;
	PyObjC_DURING
		result = CFCalendarDecomposeAbsoluteTime(
			calendar, at, componentDesc,
			&params[0], &params[1], &params[2], &params[3],
			&params[4], &params[5], &params[6], &params[7],
			&params[8], &params[9]);
	
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);

	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) {
		return NULL;
	}

	PyObject *rv = PyTuple_New(1 + strlen(componentDesc));
	if (rv == NULL) {
		return NULL;
	}

	PyObject* b = PyBool_FromLong(result);
	if (b  == NULL) {
		return NULL;
	}
	PyTuple_SetItem(rv, 0, b);

	Py_ssize_t i, len;
	len = strlen(componentDesc);
	for (i = 0; i < len; i++) {
		PyObject* v = PyInt_FromLong(params[i]);
		if (v == NULL) {
			Py_DECREF(rv);
			return NULL;
		}
		PyTuple_SetItem(rv, i+1, v);
	}
	return rv;
}

		
static PyObject*
mod_CFCalendarGetComponentDifference(
	PyObject* self __attribute__((__unused__)),
	PyObject* args)
{
	CFCalendarRef calendar;
	CFAbsoluteTime startingAt;
	CFAbsoluteTime resultAt;
	CFOptionFlags options;
	char* componentDesc;
	int params[10];
	Boolean result;
	int r;

	if (PyTuple_Size(args) < 5) {
		PyErr_Format(PyExc_TypeError, 
			"Expecting at least 5 arguments, got %" 
			PY_FORMAT_SIZE_T "d", PyTuple_Size(args));
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFCalendarRef), 
		PyTuple_GetItem(args, 0), &calendar);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFAbsoluteTime), 
		PyTuple_GetItem(args, 1), &startingAt);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFAbsoluteTime), 
		PyTuple_GetItem(args, 2), &resultAt);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(CFOptionFlags), 
		PyTuple_GetItem(args, 3), &options);
	if (r == -1) {
		return NULL;
	}

	r = PyObjC_PythonToObjC(@encode(char*), 
		PyTuple_GetItem(args, 4), &componentDesc);
	if (r == -1) {
		return NULL;
	}

	if (strlen(componentDesc) > 10) {
		PyErr_SetString(PyExc_TypeError,
			"At most 10 characters supported in componentDesc");
		return NULL;
	}

	if (PyTuple_Size(args) != 5) {
		if (PyTuple_Size(args) != 5 + strlen(componentDesc)) {
			PyErr_Format(PyExc_TypeError, 
				"Expecting %" PY_FORMAT_SIZE_T "d arguments, got %"
				PY_FORMAT_SIZE_T "d", 3 + strlen(componentDesc),
				PyTuple_Size(args));
			return NULL;
		}

		Py_ssize_t i, len;

		len = strlen(componentDesc);
		for (i = 0; i < len; i++) {
			if (PyTuple_GetItem(args, 5 + i) != Py_None) {
				PyErr_SetString(PyExc_ValueError,
					"Bad placeholder value");
				return NULL;
			}
		}
	}

	result = FALSE;
	PyObjC_DURING
		result = CFCalendarGetComponentDifference(
			calendar, startingAt, resultAt, options, 
			componentDesc,
			&params[0], &params[1], &params[2], &params[3],
			&params[4], &params[5], &params[6], &params[7],
			&params[8], &params[9]);
	
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);

	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) {
		return NULL;
	}

	PyObject *rv = PyTuple_New(1 + strlen(componentDesc));
	if (rv == NULL) {
		return NULL;
	}

	PyObject* b = PyBool_FromLong(result);
	if (b  == NULL) {
		return NULL;
	}
	PyTuple_SetItem(rv, 0, b);

	Py_ssize_t i, len;
	len = strlen(componentDesc);
	for (i = 0; i < len; i++) {
		PyObject* v = PyInt_FromLong(params[i]);
		if (v == NULL) {
			Py_DECREF(rv);
			return NULL;
		}
		PyTuple_SetItem(rv, i+1, v);
	}
	return rv;
}

#define COREFOUNDATION_CALENDAR_METHODS \
        {	\
		"CFCalendarAddComponents",	\
		(PyCFunction)mod_CFCalendarAddComponents,	\
		METH_VARARGS,	\
		NULL	\
	},	\
        {	\
		"CFCalendarComposeAbsoluteTime",	\
		(PyCFunction)mod_CFCalendarComposeAbsoluteTime,	\
		METH_VARARGS,	\
		NULL	\
	},	\
        {	\
		"CFCalendarDecomposeAbsoluteTime",	\
		(PyCFunction)mod_CFCalendarDecomposeAbsoluteTime,	\
		METH_VARARGS,	\
		NULL	\
	},	\
        {	\
		"CFCalendarGetComponentDifference",	\
		(PyCFunction)mod_CFCalendarGetComponentDifference,	\
		METH_VARARGS,	\
		NULL	\
	},
