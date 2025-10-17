/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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


static void* 
mod_writestream_retain(void* info) 
{
	PyGILState_STATE state = PyGILState_Ensure();
	Py_INCREF((PyObject*)info);
	PyGILState_Release(state);
	return info;
}

static void
mod_writestream_release(void* info)
{
	PyGILState_STATE state = PyGILState_Ensure();
	Py_DECREF((PyObject*)info);
	PyGILState_Release(state);
}


static CFStreamClientContext mod_CFStreamClientContext_Write = {
	0,		
	NULL,
	mod_writestream_retain,
	mod_writestream_release,
	NULL
};

static void
mod_CFWriteStreamClientCallBack(	
	CFWriteStreamRef f,
	CFStreamEventType eventType,
	void* _info)
{
	PyObject* info = (PyObject*)_info;
	PyGILState_STATE state = PyGILState_Ensure();

	PyObject* py_f = PyObjC_ObjCToPython(@encode(CFWriteStreamRef), &f);
	PyObject* py_eventType = PyObjC_ObjCToPython(
		@encode(CFStreamEventType), &eventType);

	PyObject* result = PyObject_CallFunction(
		PyTuple_GetItem(info, 0),
		"NNO", py_f, py_eventType, PyTuple_GetItem(info, 1));
	if (result == NULL) {
		PyObjCErr_ToObjCWithGILState(&state);
	}
	Py_DECREF(result);
	PyGILState_Release(state);
}

static PyObject*
mod_CFWriteStreamSetClient(
	PyObject* self __attribute__((__unused__)),
	PyObject* args)
{
	PyObject* py_stream;
	PyObject* py_streamEvents;
	PyObject* callout;
	PyObject* info;
	CFWriteStreamRef stream;
	CFOptionFlags streamEvents;

	if (!PyArg_ParseTuple(args, "OOOO", &py_stream, &py_streamEvents, &callout, &info)) {
		return NULL;
	}

	if (PyObjC_PythonToObjC(@encode(CFWriteStreamRef), py_stream, &stream) < 0) {
		return NULL;
	}
	if (PyObjC_PythonToObjC(@encode(CFOptionFlags), py_streamEvents, &streamEvents) < 0) {
		return NULL;
	}

	CFStreamClientContext context = mod_CFStreamClientContext_Write;
	context.info = Py_BuildValue("OO", callout, info);
	if (context.info == NULL) {
		return NULL;
	}

	Boolean rv = FALSE;
	PyObjC_DURING
		if (callout == Py_None) {
			rv = CFWriteStreamSetClient(
				stream, streamEvents, 
				NULL, &context);
		} else {
			rv = CFWriteStreamSetClient(
				stream, streamEvents, 
				mod_CFWriteStreamClientCallBack, &context);
		}

		
	PyObjC_HANDLER
		rv = FALSE;
		PyObjCErr_FromObjC(localException);

	PyObjC_ENDHANDLER

	Py_DECREF((PyObject*)context.info);
	if (PyErr_Occurred()) {
		return NULL;
	}

	return PyBool_FromLong(rv);
}

#define COREFOUNDATION_WRITESTREAM_METHODS \
        {	\
		"CFWriteStreamSetClient",	\
		(PyCFunction)mod_CFWriteStreamSetClient,	\
		METH_VARARGS,	\
		NULL	\
	},
