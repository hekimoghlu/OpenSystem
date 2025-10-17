/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

@interface PyObjC_TestOutputInitializer: NSObject
{
    int _priv;
}

-(instancetype)initWithBooleanOutput:(BOOL *)outBool;
-(BOOL)isInitialized;
@end

@implementation PyObjC_TestOutputInitializer
-(instancetype)initWithBooleanOutput:(BOOL *)outBool
{
    self = [self init];
    *outBool = YES;
    _priv = YES;
    return self;
}

-(BOOL)isInitialized
{
    return _priv;
}
@end

/* Python glue */
static PyMethodDef mod_methods[] = {
	{ 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"testoutputinitializer",
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

PyObject* PyInit_testoutputinitializer(void);

PyObject*
PyInit_testoutputinitializer(void)

#else

#define INITERROR() return
#define INITDONE() return

void inittestoutputinitializer(void);

void
inittestoutputinitializer(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("testoutputinitializer", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	if (PyModule_AddObject(m, "PyObjC_TestOutputInitializer", 
		PyObjCClass_New([PyObjC_TestOutputInitializer class])) < 0) {
		INITERROR();
	}
	INITDONE();
}
