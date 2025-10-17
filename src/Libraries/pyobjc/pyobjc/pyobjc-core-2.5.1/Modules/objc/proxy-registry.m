/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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

static NSMapTable* python_proxies = NULL;
static NSMapTable* objc_proxies = NULL;

int PyObjC_InitProxyRegistry(void)
{
	python_proxies = NSCreateMapTable(
			PyObjCUtil_PointerKeyCallBacks,
			PyObjCUtil_PointerValueCallBacks,
			0);
	if (python_proxies == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
			"Cannot create NSMapTable for python_proxies");
		return -1;
	}

	objc_proxies = NSCreateMapTable(
			PyObjCUtil_PointerKeyCallBacks,
			PyObjCUtil_PointerValueCallBacks,
			0);
	if (objc_proxies == NULL) {
		PyErr_SetString(PyExc_RuntimeError,
			"Cannot create NSMapTable for objc_proxies");
		return -1;
	}
	return 0;
}

int PyObjC_RegisterPythonProxy(id original, PyObject* proxy)
{
	NSMapInsert(python_proxies, original, proxy);
	return 0;
}

int PyObjC_RegisterObjCProxy(PyObject* original, id proxy)
{
	NSMapInsert(objc_proxies, original, proxy);
	return 0;
}

void PyObjC_UnregisterPythonProxy(id original, PyObject* proxy)
{
	PyObject* v;

	if (original == nil) return;

	v = NSMapGet(python_proxies, original);
	if (v == proxy) {
		NSMapRemove(python_proxies, original);
	}
}

void PyObjC_UnregisterObjCProxy(PyObject* original, id proxy)
{
	id v;

	if (original == NULL) return;

	v = NSMapGet(objc_proxies, original);
	if (v == proxy) {
		NSMapRemove(objc_proxies, original);
	}
}

PyObject* PyObjC_FindPythonProxy(id original)
{
	PyObject* v;
	
	if (original == nil) {
		v = Py_None;
	} else {
		v = NSMapGet(python_proxies, original);
	}
	Py_XINCREF(v);
	return v;
}

id PyObjC_FindObjCProxy(PyObject* original)
{
    if (original == Py_None) {
        return nil;
    } else {
        return NSMapGet(objc_proxies, original);
    }
}
