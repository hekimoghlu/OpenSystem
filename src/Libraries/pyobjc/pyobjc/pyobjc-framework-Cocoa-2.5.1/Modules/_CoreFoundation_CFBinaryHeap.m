/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 9, 2024.
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
@interface NSObject (OC_Comparison)
-(NSComparisonResult)compare:(NSObject*)other;
@end


static const void *mod_binheap_retain (CFAllocatorRef allocator __attribute__((__unused__)), const void *ptr)
{
	if (ptr) {
		CFRetain(ptr);
	}
	return ptr;
}

static void mod_binheap_release (CFAllocatorRef allocator __attribute__((__unused__)), const void *ptr)
{
	if (ptr) {
		CFRelease(ptr);
	}
}

static CFStringRef mod_binheap_copydescription(const void* ptr)
{
	CFStringRef r =  CFCopyDescription(ptr);	
	return r;
}

CFComparisonResult mod_binheap_compare(const void *ptr1, const void *ptr2, void *info __attribute__((__unused__)))
{
	NSObject* o1 = (NSObject*)ptr1;
	NSObject* o2 = (NSObject*)ptr2;


	NSComparisonResult result = [o1 compare:o2];
	return (CFComparisonResult)result;
}


static CFBinaryHeapCallBacks mod_NSObjectBinaryHeapCallbacks = {
	0, 
	mod_binheap_retain,
	mod_binheap_release,
	mod_binheap_copydescription,
	mod_binheap_compare
};


static PyObject* 
mod_CFBinaryHeapCreate(PyObject* self __attribute__((__unused__)),
	PyObject* args)
{
	PyObject* py_allocator;
	Py_ssize_t count = -1;
	CFAllocatorRef allocator;
	CFBinaryHeapRef heap;


	if (!PyArg_ParseTuple(args, "O" Py_ARG_SIZE_T, &py_allocator, &count)) {
		return NULL;
	}

	if (PyObjC_PythonToObjC(@encode(CFAllocatorRef), py_allocator, &allocator) < 0) {
		return NULL;
	}

	heap = CFBinaryHeapCreate(allocator, count, &mod_NSObjectBinaryHeapCallbacks, NULL);

	PyObject* result = PyObjC_ObjCToPython(@encode(CFBinaryHeapRef), &heap);
	if (heap) {
		CFRelease(heap);
	}
	return result;
}


static PyObject*
mod_CFBinaryHeapGetValues(
	PyObject* self __attribute__((__unused__)), 
	PyObject* args)
{
	PyObject* py_heap;
	CFBinaryHeapRef heap;


	if (!PyArg_ParseTuple(args, "O", &py_heap)) {
		return NULL;
	}

	if (PyObjC_PythonToObjC(@encode(CFBinaryHeapRef), py_heap, &heap) < 0) {
		return NULL;
	}

	CFIndex count = CFBinaryHeapGetCount(heap);
	NSObject** members = malloc(sizeof(NSObject*) * count);
	if (members == NULL) {
		PyErr_NoMemory();
		return NULL;
	}
	memset(members, 0, sizeof(NSObject*) * count);

	CFBinaryHeapGetValues(heap, (const void**)members);
	PyObject* result = PyObjC_CArrayToPython(@encode(NSObject*), members, (Py_ssize_t)count);
	free(members);
	return result;
}

#define COREFOUNDATION_CFBINARYHEAP_METHODS \
        { \
		"CFBinaryHeapCreate", \
		(PyCFunction)mod_CFBinaryHeapCreate, \
		METH_VARARGS, \
		NULL \
	}, \
        { \
		"CFBinaryHeapGetValues", \
		(PyCFunction)mod_CFBinaryHeapGetValues, \
		METH_VARARGS, \
		NULL \
	},
