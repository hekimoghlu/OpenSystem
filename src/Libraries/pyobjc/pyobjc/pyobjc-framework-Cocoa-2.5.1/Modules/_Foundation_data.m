/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 13, 2023.
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

static PyObject* call_NSData_bytes(
	PyObject* method, PyObject* self, PyObject* arguments)
{
	const void* bytes;
	NSUInteger    bytes_len;
	PyObject* result;
	struct objc_super super;

	if (!PyArg_ParseTuple(arguments, "")) {
		return NULL;
	}

	PyObjC_DURING
		PyObjC_InitSuper(&super,
			PyObjCSelector_GetClass(method),
			PyObjCObject_GetObject(self));

		bytes = ((void*(*)(struct objc_super*, SEL))objc_msgSendSuper)(&super, 
				PyObjCSelector_GetSelector(method));
		bytes_len = ((NSUInteger(*)(struct objc_super*, SEL))objc_msgSendSuper)(&super, @selector(length));


	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		result = NULL;
		bytes = NULL;
		bytes_len = 0;
	PyObjC_ENDHANDLER

	if (bytes == NULL && PyErr_Occurred()) return NULL;

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION <= 6
	result = PyBuffer_FromMemory((char*)bytes, bytes_len);
#else
	/* 2.7 or later: use a memory view */
	Py_buffer info;
	if (PyBuffer_FillInfo(&info, self, (void*)bytes, bytes_len, 1, PyBUF_FULL_RO) < 0) {
		return NULL;
	}
	result = PyMemoryView_FromBuffer(&info);
#endif

	return result;
}

static void 
imp_NSData_bytes(
	void* cif __attribute__((__unused__)), 
	void* resp, 
	void** args, 
	void* callable)
{
	id self = *(id*)args[0];
	//SEL _meth = *(SEL*)args[1];
	void** pretval = (void**)resp;

	PyObject* result;
	PyObject* arglist = NULL;
	PyObject* pyself = NULL;
	int cookie = 0;

	PyGILState_STATE state = PyGILState_Ensure();

	arglist = PyTuple_New(1);
	if (arglist == NULL) goto error;

	pyself = PyObjCObject_NewTransient(self, &cookie);
	if (pyself == NULL) goto error;
	PyTuple_SetItem(arglist, 0, pyself); 
	Py_INCREF(pyself);

	result = PyObject_Call((PyObject*)callable, arglist, NULL);
	Py_DECREF(arglist); arglist = NULL;
	PyObjCObject_ReleaseTransient(pyself, cookie); pyself = NULL;
	if (result == NULL) goto error;

	if (result == Py_None) {
		*pretval = NULL;
		Py_DECREF(result);
		PyGILState_Release(state);
		return;
	}

#if PY_MAJOR_VERSION == 2
	if (PyBuffer_Check(result)) {
		/* XXX: Is this correct?? */
		const void *p;
		Py_ssize_t len;
		if (PyObject_AsReadBuffer(result, &p, &len) == -1) {
			goto error;
		}
		Py_DECREF(result);
		*pretval =  (void *)p;
		PyGILState_Release(state);
		return;
	} else 
#endif
	if (PyBytes_Check(result)) {
		/* XXX: Is this correct */
		void* p;

		p = PyBytes_AsString(result);
		*pretval = (void*)p;
		PyGILState_Release(state);
		return;
	}

	PyErr_SetString(PyExc_ValueError, "No idea what to do with result.");
	goto error;

error:
	Py_XDECREF(arglist);
	if (pyself) {
		PyObjCObject_ReleaseTransient(pyself, cookie); 
	}
	PyObjCErr_ToObjCWithGILState(&state);
	*pretval = NULL;
}


static PyObject* 
call_NSMutableData_mutableBytes(
	PyObject* method, PyObject* self, PyObject* arguments)
{
	void*     bytes;
	NSUInteger  bytes_len;
	PyObject* result;
	struct objc_super super;

	if (!PyArg_ParseTuple(arguments, "")) {
		return NULL;
	}

	PyObjC_DURING
		PyObjC_InitSuper(&super,
			PyObjCSelector_GetClass(method),
			PyObjCObject_GetObject(self));

		bytes = ((void*(*)(struct objc_super*, SEL))objc_msgSendSuper)(&super, 
				PyObjCSelector_GetSelector(method));
		bytes_len = ((NSUInteger(*)(struct objc_super*,SEL))objc_msgSendSuper)(&super, @selector(length));

	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		result = NULL;
		bytes = NULL;
		bytes_len = 0;
	PyObjC_ENDHANDLER

	if (bytes == NULL && PyErr_Occurred()) return NULL;

#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION <= 6
	result = PyBuffer_FromReadWriteMemory((void*)bytes, bytes_len);

#else
	/* 2.7 or later: use a memory view */
	Py_buffer info;
	if (PyBuffer_FillInfo(&info, self, bytes, bytes_len, 0, PyBUF_FULL) < 0) {
		return NULL;
	}
	result = PyMemoryView_FromBuffer(&info);
#endif

	return result;
}

static void
imp_NSMutableData_mutableBytes(
	void* cif __attribute__((__unused__)), 
	void* resp, 
	void** args, 
	void* callable)
{
	id self = *(id*)args[0];
	//SEL _meth = *(SEL*)args[1];
	void** pretval = (void**)resp;
	PyObject* result;
	PyObject* arglist = NULL;
	PyObject* pyself = NULL;
	int cookie = 0;

	PyGILState_STATE state = PyGILState_Ensure();

	arglist = PyTuple_New(1);
	if (arglist == NULL) goto error;

	pyself = PyObjCObject_NewTransient(self, &cookie);
	if (pyself == NULL) goto error;
	PyTuple_SetItem(arglist, 0, pyself); 
	Py_INCREF(pyself);

	result = PyObject_Call((PyObject*)callable, arglist, NULL);
	Py_DECREF(arglist); arglist = NULL;
	PyObjCObject_ReleaseTransient(pyself, cookie); pyself = NULL;
	if (result == NULL) goto error;

	if (result == Py_None) {
		Py_DECREF(result);
		goto error;
	}

	if (result == Py_None) {
		*pretval = NULL;
		Py_DECREF(result);
		PyGILState_Release(state);
		return;
	}

	void *p;
	Py_ssize_t len;
	if (PyObject_AsWriteBuffer(result, &p, &len) == -1) goto error;
	Py_DECREF(result);
	*pretval = (void *)p;
	PyGILState_Release(state);
	return;

error:
	Py_XDECREF(arglist);
	if (pyself) {
		PyObjCObject_ReleaseTransient(pyself, cookie); 
	}
	*pretval = NULL;
	PyObjCErr_ToObjCWithGILState(&state);
}

static int setup_nsdata(PyObject* m __attribute__((__unused__)))
{
	Class classNSData = objc_lookUpClass("NSData");
	Class classNSMutableData = objc_lookUpClass("NSMutableData");

	if (classNSData != NULL) {

		if (PyObjC_RegisterMethodMapping(classNSData, 
				 @selector(bytes),
				 call_NSData_bytes,
				 imp_NSData_bytes) < 0 ) {
			return -1;
		}

	}

	if (classNSMutableData != NULL) {

		if (PyObjC_RegisterMethodMapping(classNSMutableData, 
				@selector(mutableBytes),
				call_NSMutableData_mutableBytes,
				imp_NSMutableData_mutableBytes) < 0 ) {
			return -1;
		}
	}
  
	return 0;
}
