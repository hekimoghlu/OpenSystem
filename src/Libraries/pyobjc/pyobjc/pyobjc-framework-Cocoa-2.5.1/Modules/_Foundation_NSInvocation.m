/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
call_NSInvocation_setArgument_atIndex_(
	PyObject* method, PyObject* self, PyObject* arguments)
{
	struct objc_super super;
	NSMethodSignature* signature;
	const char* tp;
	PyObject* py_value;
	NSUInteger index;
	void* buf;
	Py_ssize_t sz;

	if  (!PyArg_ParseTuple(arguments, "O" Py_ARG_NSUInteger, &py_value, &index)) {
		return NULL;
	}

	PyObjC_DURING
		signature = [(NSInvocation*)PyObjCObject_GetObject(self) methodSignature];

		tp = [signature getArgumentTypeAtIndex:index];
	
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		signature = NULL;
		tp = NULL;

	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) {
		return NULL;
	}

	sz = PyObjCRT_SizeOfType(tp);
	if (sz == -1) {
		return NULL;
	}

	buf = PyMem_Malloc(sz);
	if (buf == NULL) {
		PyErr_NoMemory();
		return NULL;
	}

	if (PyObjC_PythonToObjC(tp, py_value, buf) == -1) {
		PyMem_Free(buf);
		return NULL;
	}

	PyObjC_DURING
		if (PyObjCIMP_Check(method)) {
			((void(*)(id,SEL, void*, NSUInteger))
				(PyObjCIMP_GetIMP(method)))(
					PyObjCObject_GetObject(self),
					PyObjCIMP_GetSelector(method),
					buf, index);
		} else {
			PyObjC_InitSuper(&super, 
				PyObjCSelector_GetClass(method),
				PyObjCObject_GetObject(self));

			((void(*)(struct objc_super*, SEL, void*, NSUInteger))objc_msgSendSuper)(&super,
					PyObjCSelector_GetSelector(method),
					buf, index);
		}
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
	PyObjC_ENDHANDLER

	PyMem_Free(buf);

	if (PyErr_Occurred()) {
		return NULL;
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* 
call_NSInvocation_setReturnValue_(
	PyObject* method, PyObject* self, PyObject* arguments)
{
	struct objc_super super;
	NSMethodSignature* signature;
	const char* tp;
	PyObject* py_value;
	void* buf;
	Py_ssize_t sz;

	if  (!PyArg_ParseTuple(arguments, "O", &py_value)) {
		return NULL;
	}

	PyObjC_DURING
		signature = [(NSInvocation*)PyObjCObject_GetObject(self) methodSignature];

		tp = [signature methodReturnType];
	
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		signature = NULL;
		tp = NULL;

	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) {
		return NULL;
	}

	sz = PyObjCRT_SizeOfType(tp);
	if (sz == -1) {
		return NULL;
	}

	buf = PyMem_Malloc(sz);
	if (buf == NULL) {
		PyErr_NoMemory();
		return NULL;
	}

	if (PyObjC_PythonToObjC(tp, py_value, buf) == -1) {
		PyMem_Free(buf);
		return NULL;
	}

	PyObjC_DURING
		if (PyObjCIMP_Check(method)) {
			((void(*)(id,SEL, void*))
				(PyObjCIMP_GetIMP(method)))(
					PyObjCObject_GetObject(self),
					PyObjCIMP_GetSelector(method),
					buf);
		} else {
			PyObjC_InitSuper(&super, 
				PyObjCSelector_GetClass(method),
				PyObjCObject_GetObject(self));

			((void(*)(struct objc_super*, SEL, void*))objc_msgSendSuper)(&super,
					PyObjCSelector_GetSelector(method),
					buf);
		}
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
	PyObjC_ENDHANDLER

	PyMem_Free(buf);

	if (PyErr_Occurred()) {
		return NULL;
	}

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* 
call_NSInvocation_getArgument_atIndex_(
	PyObject* method, PyObject* self, PyObject* arguments)
{
	struct objc_super super;
	NSMethodSignature* signature;
	const char* tp;
	PyObject* py_value;
	NSUInteger index;
	void* buf;
	Py_ssize_t sz;

	if  (!PyArg_ParseTuple(arguments, "O" Py_ARG_NSUInteger, &py_value, &index)) {
		return NULL;
	}

	if (py_value != Py_None) {
		PyErr_SetString(PyExc_ValueError, "buffer must be None");
		return NULL;
	}

	PyObjC_DURING
		signature = [(NSInvocation*)PyObjCObject_GetObject(self) methodSignature];

		tp = [signature getArgumentTypeAtIndex:index];
	
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		signature = NULL;
		tp = NULL;

	PyObjC_ENDHANDLER


	if (PyErr_Occurred()) {
		return NULL;
	}

	sz = PyObjCRT_SizeOfType(tp);
	if (sz == -1) {
		return NULL;
	}

	buf = PyMem_Malloc(sz);
	if (buf == NULL) {
		PyErr_NoMemory();
		return NULL;
	}


	PyObjC_DURING
		if (PyObjCIMP_Check(method)) {
			((void(*)(id,SEL, void*, NSUInteger))
				(PyObjCIMP_GetIMP(method)))(
					PyObjCObject_GetObject(self),
					PyObjCIMP_GetSelector(method),
					buf, index);
		} else {
			PyObjC_InitSuper(&super, 
				PyObjCSelector_GetClass(method),
				PyObjCObject_GetObject(self));

			((void(*)(struct objc_super*,SEL,void*,NSUInteger))objc_msgSendSuper)(&super,
					PyObjCSelector_GetSelector(method),
					buf, index);
		}
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
	PyObjC_ENDHANDLER


	if (PyErr_Occurred()) {
		PyMem_Free(buf);
		return NULL;
	}

	py_value = PyObjC_ObjCToPython(tp, buf);
	PyMem_Free(buf);
	if (py_value == NULL) {
		return NULL;
	}

	return py_value;
}

static PyObject* 
call_NSInvocation_getReturnValue_(
	PyObject* method, PyObject* self, PyObject* arguments)
{
	struct objc_super super;
	NSMethodSignature* signature;
	const char* tp;
	PyObject* py_value;
	void* buf;
	Py_ssize_t sz;

	if  (!PyArg_ParseTuple(arguments, "O", &py_value)) {
		return NULL;
	}

	if (py_value != Py_None) {
		PyErr_SetString(PyExc_ValueError, "buffer must be None");
		return NULL;
	}

	PyObjC_DURING
		signature = [(NSInvocation*)PyObjCObject_GetObject(self) methodSignature];

		tp = [signature methodReturnType];
	
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
		signature = NULL;
		tp = NULL;

	PyObjC_ENDHANDLER


	if (PyErr_Occurred()) {
		return NULL;
	}

	sz = PyObjCRT_SizeOfType(tp);
	if (sz == -1) {
		return NULL;
	}

	buf = PyMem_Malloc(sz);
	if (buf == NULL) {
		PyErr_NoMemory();
		return NULL;
	}


	PyObjC_DURING
		if (PyObjCIMP_Check(method)) {
			((void(*)(id,SEL, void*))
				(PyObjCIMP_GetIMP(method)))(
					PyObjCObject_GetObject(self),
					PyObjCIMP_GetSelector(method),
					buf);
		} else {
			PyObjC_InitSuper(&super, 
				PyObjCSelector_GetClass(method),
				PyObjCObject_GetObject(self));

			((void(*)(struct objc_super*, SEL, void*))objc_msgSendSuper)(&super,
					PyObjCSelector_GetSelector(method),
					buf);
		}
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
	PyObjC_ENDHANDLER


	if (PyErr_Occurred()) {
		PyMem_Free(buf);
		return NULL;
	}

	py_value = PyObjC_ObjCToPython(tp, buf);
	PyMem_Free(buf);
	if (py_value == NULL) {
		return NULL;
	}

	return py_value;
}


static int setup_nsinvocation(PyObject* m __attribute__((__unused__)))
{
	Class classNSInvocation = objc_lookUpClass("NSInvocation");
  
	if (PyObjC_RegisterMethodMapping(
			classNSInvocation,
			@selector(setArgument:atIndex:),
			call_NSInvocation_setArgument_atIndex_,
			PyObjCUnsupportedMethod_IMP) < 0) {
		return -1;
	}

	if (PyObjC_RegisterMethodMapping(
			classNSInvocation,
			@selector(setReturnValue:),
			call_NSInvocation_setReturnValue_,
			PyObjCUnsupportedMethod_IMP) < 0) {
		return -1;
	}

	if (PyObjC_RegisterMethodMapping(
			classNSInvocation,
			@selector(getArgument:atIndex:),
			call_NSInvocation_getArgument_atIndex_,
			PyObjCUnsupportedMethod_IMP) < 0) {
		return -1;
	}

	if (PyObjC_RegisterMethodMapping(
			classNSInvocation,
			@selector(getReturnValue:),
			call_NSInvocation_getReturnValue_,
			PyObjCUnsupportedMethod_IMP) < 0) {
		return -1;
	}

	return 0;
}
