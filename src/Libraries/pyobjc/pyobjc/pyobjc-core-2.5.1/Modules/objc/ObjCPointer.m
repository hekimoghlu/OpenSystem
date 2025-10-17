/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
#include "pyobjc.h"

int PyObjCPointer_RaiseException = 0;

static void
PyObjCPointer_dealloc (PyObject* _self)
{
	PyObjCPointer* self = (PyObjCPointer*)_self;
	Py_DECREF (self->type);
	PyObject_Free((PyObject*)self);
}

PyDoc_STRVAR(PyObjCPointer_unpack_doc,
	"Unpack the pointed value accordingly to its type.\n"
        "obj.unpack() -> value");
static PyObject *
PyObjCPointer_unpack (PyObject* _self)
{
	PyObjCPointer* self = (PyObjCPointer*)_self;

	if (self->ptr) {
		const char *type = PyBytes_AS_STRING (self->type);

		if (*type == _C_VOID) {
			PyErr_SetString (PyObjCExc_Error, 
				"Cannot dereference a pointer to void");
			return NULL;
		} else {
			return pythonify_c_value (type, self->ptr);
		}
        } else {
		Py_INCREF (Py_None);
		return Py_None;
        }
}

static PyMethodDef PyObjCPointer_methods[] =
{
	{
		"unpack",   
		(PyCFunction)PyObjCPointer_unpack,       
		METH_NOARGS,   
		PyObjCPointer_unpack_doc 
	},
	{ 0, 0, 0, 0 }
};

static PyMemberDef PyObjCPointer_members[] = {
	{
		"type",
		T_OBJECT,
		offsetof(PyObjCPointer, type),
		READONLY,
		NULL
	},
	{
		"pointerAsInteger",
		T_INT,
		offsetof(PyObjCPointer, ptr),
		READONLY,
		NULL
	},
	{ 0, 0, 0, 0, 0 }
};

PyTypeObject PyObjCPointer_Type =
{
	PyVarObject_HEAD_INIT(&PyType_Type, 0)
	"PyObjCPointer",			/* tp_name */
	sizeof (PyObjCPointer),			/* tp_basicsize */
	sizeof (char),				/* tp_itemsize */
  
	/* methods */
	PyObjCPointer_dealloc,			/* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	0,					/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	0,					/* tp_call */
	0,					/* tp_str */
	0,					/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,			/* tp_flags */
	"Wrapper around a Objective-C Pointer",	/* tp_doc */
	0,					/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	0,					/* tp_iter */
	0,					/* tp_iternext */
	PyObjCPointer_methods,			/* tp_methods */
	PyObjCPointer_members,			/* tp_members */
	0,					/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	0,					/* tp_init */
	0,					/* tp_alloc */
	0,					/* tp_new */
	0,					/* tp_free */
	0,					/* tp_is_gc */
	0,					/* tp_bases */
	0,					/* tp_mro */
	0,					/* tp_cache */
	0,					/* tp_subclasses */
	0,					/* tp_weaklist */
	0					/* tp_del */
#if PY_VERSION_HEX >= 0x02060000
	, 0                                     /* tp_version_tag */
#endif

};

PyObjCPointer *
PyObjCPointer_New(void *p, const char *t)
{
	Py_ssize_t size = PyObjCRT_SizeOfType (t);
	const char *typeend = PyObjCRT_SkipTypeSpec (t);
	PyObjCPointer *self;

	if (PyObjCPointer_RaiseException) {
		PyErr_Format(PyObjCExc_UnknownPointerError,
			"pointer of type %s", t);
		return NULL;
	}
	NSLog(@"PyObjCPointer created: at %p of type %s", p, t);

	if (size == -1) {
		return NULL;
	}
	if (typeend == NULL) {
		return NULL;
	}
  
	self = PyObject_NEW_VAR (PyObjCPointer, &PyObjCPointer_Type, size);
	if (self == NULL) {
		return NULL;
	}

	self->type = PyBytes_FromStringAndSize ((char *) t, typeend-t);

	if (size && p) {
		memcpy ((self->ptr = self->contents), p, size);
	} else {
		self->ptr = p;
	}
  
	return self;
}
