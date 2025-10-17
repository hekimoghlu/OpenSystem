/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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
call_NSView_getRectsBeingDrawn_count_(
	PyObject* method, PyObject* self, PyObject* arguments)
{
	PyObject* result;
	struct objc_super super;
	PyObject* v;
	NSRect* rects;
	PyObject* arg1, *arg2;
	NSInteger count;

	if  (!PyArg_ParseTuple(arguments, "OO", &arg1, &arg2)) {
		return NULL;
	}

	if (arg1 != Py_None) {
		PyErr_SetString(PyExc_ValueError, "buffer must be None");
		return NULL;
	}
	if (arg2 != Py_None) {
		PyErr_SetString(PyExc_ValueError, "count must be None");
		return NULL;
	}


	PyObjC_DURING
		PyObjC_InitSuper(&super, 
			PyObjCSelector_GetClass(method),
			PyObjCObject_GetObject(self));

			
		((void(*)(struct objc_super*, SEL, NSRect**, NSInteger*))objc_msgSendSuper)(&super,
				PyObjCSelector_GetSelector(method),
				&rects, &count);
	PyObjC_HANDLER
		PyObjCErr_FromObjC(localException);
	PyObjC_ENDHANDLER

	if (PyErr_Occurred()) return NULL;

	v = PyObjC_CArrayToPython(
#ifdef __LP64__
	"{_NSRect={_NSPoint=dd}{_NSSize=dd}}",
#else
	"{_NSRect={_NSPoint=ff}{_NSSize=ff}}",
#endif
		rects, count);
	if (v == NULL) return NULL;

	result = Py_BuildValue("Oi", v, count);
	Py_XDECREF(v);

	return result;
}



static int setup_nsview(PyObject* m __attribute__((__unused__)))
{
	Class classNSView = objc_lookUpClass("NSView");
	if (classNSView == NULL) {
		return 0;
	}

	if (PyObjC_RegisterMethodMapping(
		classNSView,
		@selector(getRectsBeingDrawn:count:),
		call_NSView_getRectsBeingDrawn_count_,
		PyObjCUnsupportedMethod_IMP) < 0) {

		return -1;
	}

	return 0;
}
