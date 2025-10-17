/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
#include <pyobjc-api.h>

#import <Foundation/Foundation.h>

#ifndef GNU_RUNTIME
#include <objc/objc-runtime.h>
#endif

@interface PyObjC_TestUnallocatable : NSObject
{
}
@end

@implementation PyObjC_TestUnallocatable
+ (id)allocWithZone:(NSZone *)zone 
{
    (void)&zone; /* Force use */
    return nil;
}
@end

@interface PyObjC_TestClassAndInstance: NSObject
{
}

+ (BOOL)isInstance;
- (BOOL)isInstance;
@end

@implementation PyObjC_TestClassAndInstance
+(BOOL)isInstance
{
	return NO;
}
-(BOOL)isInstance
{
	return YES;
}
@end

/* Python glue */
static PyMethodDef mod_methods[] = {
	{ 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"testclassandinst",
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

PyObject* PyInit_testclassandinst(void);

PyObject*
PyInit_testclassandinst(void)

#else

#define INITERROR() return
#define INITDONE() return

void inittestclassandinst(void);

void
inittestclassandinst(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("testclassandinst", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	if (PyModule_AddObject(m, "PyObjC_TestClassAndInstance", 
		PyObjCClass_New([PyObjC_TestClassAndInstance class])) < 0) {
		INITERROR();
	}

	if (PyModule_AddObject(m, "PyObjC_TestUnallocatable", 
		PyObjCClass_New([PyObjC_TestUnallocatable class])) < 0) {
		INITERROR();
	}

	INITDONE();
}
