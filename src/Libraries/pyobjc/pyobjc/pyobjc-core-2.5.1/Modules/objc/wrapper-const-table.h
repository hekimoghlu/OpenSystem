/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
#import <ApplicationServices/ApplicationServices.h>
#import <Foundation/NSString.h>
#import <Foundation/NSArray.h>

struct vartable {
	NSString* name;
	char* type;
};

struct inttable {
	char* 	name;
	int     is_unsigned;
	int	value;
};

struct stringtable {
	char*	  name;
	NSString* const* value;
};

static inline int add_double(PyObject*d, char* name, double value)
{
	int res;
	PyObject* v;

	v = PyObjC_ObjCToPython(@encode(double), &value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	Py_DECREF(v);
	if (res < 0) return -1;
	return 0;
}

#ifndef NO_OBJC2_RUNTIME
static inline int add_CGFloat(PyObject*d, char* name, double value)
{
	int res;
	PyObject* v;

	v = PyObjC_ObjCToPython(@encode(CGFloat), &value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	Py_DECREF(v);
	if (res < 0) return -1;
	return 0;
}
#endif


static inline int add_float(PyObject*d, char* name, float value)
{
	int res;
	PyObject* v;

	v = PyObjC_ObjCToPython(@encode(float), &value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	Py_DECREF(v);
	if (res < 0) return -1;
	return 0;
}

static inline int add_unsigned(PyObject*d, char* name, unsigned value)
{
	int res;
	PyObject* v;

	v = PyObjC_ObjCToPython(@encode(unsigned), &value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	Py_DECREF(v);
	if (res < 0) return -1;
	return 0;
}

static inline int add_BOOL(PyObject*d, char* name, BOOL value)
{
	int res;
	PyObject* v;

	v = PyBool_FromLong(value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	Py_DECREF(v);
	if (res < 0) return -1;
	return 0;
}

static inline int add_int(PyObject*d, char* name, int value)
{
	int res;
	PyObject* v;

	v = PyObjC_ObjCToPython(@encode(int), &value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	Py_DECREF(v);
	if (res < 0) return -1;
	return 0;
}
#ifndef NO_OBJC2_RUNTIME
static inline int add_NSUInteger(PyObject*d, char* name, NSUInteger value)
{
	int res;
	PyObject* v;

	v = PyObjC_ObjCToPython(@encode(NSUInteger), &value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	Py_DECREF(v);
	if (res < 0) return -1;
	return 0;
}
#endif

static inline int register_ints(PyObject* d, struct inttable* table)
{
	while (table->name != NULL) {
		if (table->is_unsigned) {
			int res = add_unsigned(d, table->name, 
					(unsigned)table->value);
			if (res == -1) return -1;
		} else {
			int res = add_int(d, table->name, table->value);
			if (res == -1) return -1;
		}

		table++;
	}
	return 0;
}

static inline int add_string(PyObject* d, char* name, NSString* value)
{
	int res;
	PyObject* v;

	v = PyObjC_ObjCToPython(@encode(id), &value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	if (res < 0) return -1;
	return 0;
}

static inline int add_id(PyObject* d, char* name, id value)
{
	int res;
	PyObject* v;

	v = PyObjC_ObjCToPython(@encode(id), &value);
	if (v == NULL) return -1;

	res = PyDict_SetItemString(d, name, v);
	if (res < 0) return -1;
	return 0;
}


static inline int register_strings(PyObject* d, struct stringtable* table)
{
	while (table->name != NULL) {
		add_string(d, table->name, *table->value);
		table++;
	}
	return 0;
}

#import <CoreFoundation/CoreFoundation.h>

static inline int
register_variableList(PyObject* d, CFBundleRef bundle __attribute__((__unused__)), struct vartable* table, size_t count)
{
	void** ptrs = NULL;
	NSMutableArray* names = nil;
	size_t i;
	int retVal = 0;

	ptrs = PyMem_Malloc(sizeof(void*) * count);
	if (ptrs == NULL) {
		PyErr_NoMemory();
		return -1;
	}

	names = [[NSMutableArray alloc] init];
	if (names == NULL) {
		PyErr_NoMemory();
		retVal = -1;
		goto cleanup;
	}

	for (i = 0; i < count; i++) {
		[names addObject:table[i].name];
	}

	CFBundleGetDataPointersForNames(bundle,
		(CFArrayRef)names, ptrs);

	for (i = 0; i < count; i++) {
		PyObject* val;
		if (ptrs[i] == NULL) continue; /* Skip undefined names */

		val  = PyObjC_ObjCToPython(table[i].type, ptrs[i]);
		if (val == NULL) {
			retVal = -1;
			goto cleanup;
		}
		PyDict_SetItemString(d, (char*)[table[i].name cString], val);
		Py_DECREF(val);
	}

cleanup:
	if (ptrs) {
		PyMem_Free(ptrs);
	}

	[names release];

	return retVal;
}
