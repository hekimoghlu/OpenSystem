/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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

#include <Python.h>
#include "pyobjc-api.h"

#import <Foundation/Foundation.h>

typedef struct s { int i; char b; } struct_s;
@interface OCPropertyDefinitions : NSObject {
	int _prop1;
	float _prop2;
	struct_s _prop3;
	id	_prop4;
	id	_prop5;
	id	_prop6;
	id	_prop7;
	id	_prop8;
	id	_prop9;
	struct_s _prop10;
	id	_prop11;
	id	_prop12;
}

#if (PyObjC_BUILD_RELEASE >= 1005) 

#pragma message "Ignore warnings about properties in this file."
@property int prop1;
@property float prop2;
@property struct_s prop3;
@property id prop4;
@property(readonly) id prop5;
@property(readwrite) id prop6;
@property(assign) id prop7;
@property(retain) id prop8;
@property(copy) id prop9;
@property(nonatomic) struct_s prop10;
@property(getter=propGetter,setter=propSetter:) id prop11;
@property(nonatomic,readwrite,retain) id prop12;
@property(readwrite,copy) id prop13;

#endif

@end

@implementation OCPropertyDefinitions

#if (PyObjC_BUILD_RELEASE >= 1005 )

@synthesize prop1 = _prop1;
@synthesize prop2 = _prop2;
@synthesize prop3 = _prop3;
@synthesize prop4 = _prop4;
@synthesize prop5 = _prop5;
@synthesize prop6 = _prop6;
@synthesize prop7 = _prop7;
@synthesize prop8 = _prop8;
@synthesize prop9 = _prop9;
@synthesize prop10 = _prop10;
@synthesize prop11 = _prop11;
@synthesize prop12 = _prop12;
@dynamic prop13;

#endif

@end


static PyMethodDef mod_methods[] = {
	        { 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"properties",
	NULL,
	0,
	mod_methods,
	NULL,
	NULL,
	NULL,
	NULL
};

#define INITERROR() return NULL
#define INITDONE() return m

PyObject* PyInit_properties(void);

PyObject*
PyInit_properties(void)

#else

#define INITERROR() return
#define INITDONE() return

void initproperties(void);

void
initproperties(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("properties", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	if (PyModule_AddObject(m, "OCPropertyDefinitions",
	    PyObjCClass_New([OCPropertyDefinitions class])) < 0) {
		INITERROR();
	}

	INITDONE();
}
