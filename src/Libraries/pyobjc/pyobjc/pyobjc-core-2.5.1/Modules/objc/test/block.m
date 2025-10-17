/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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

@interface OCTestBlock : NSObject {}

#if (PyObjC_BUILD_RELEASE >= 1006) && (__GNUC__ >= 4 && __GNUC_MINOR__ >= 2)

-(int(^)(void))getIntBlock;
-(double(^)(double,double))getFloatBlock;
-(void)callIntBlock:(void(^)(int))block withValue:(int)value;
-(double)callDoubleBlock:(double(^)(double, double))block withValue:(double)v1 andValue:(double)v2;
#endif

@end

@implementation OCTestBlock

#if PyObjC_BUILD_RELEASE >= 1006 && (__GNUC__ >= 4 && __GNUC_MINOR__ >= 2)

-(int(^)(void))getIntBlock
{
	return [[^{ return 42; } copy] autorelease];
}

-(double(^)(double,double))getFloatBlock
{
	return [[^(double a, double b) { return a + b; } copy] autorelease];
}

-(void)callIntBlock:(void(^)(int))block withValue:(int)value
{
	block(value);
}

-(double)callDoubleBlock: (double(^)(double, double))block withValue:(double)v1 andValue:(double)v2
{
	return block(v1, v2);
}
#endif

@end


static PyMethodDef mod_methods[] = {
	        { 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"block",
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

PyObject* PyInit_block(void);

PyObject*
PyInit_block(void)

#else

#define INITERROR() return
#define INITDONE() return

void initblock(void);

void
initblock(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("block", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	if (PyModule_AddObject(m, "OCTestBlock",
	    PyObjCClass_New([OCTestBlock class])) < 0) {
		INITERROR();
	}

	INITDONE();
}
