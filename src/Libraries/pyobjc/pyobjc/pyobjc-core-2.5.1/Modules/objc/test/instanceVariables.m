/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
#include "Python.h"
#include "pyobjc-api.h"

#import <Foundation/Foundation.h>

@interface ClassWithVariables : NSObject
{ 
	int	intValue;
	double	floatValue;
	char	charValue;
	char*	strValue;
	NSRect  rectValue;
	NSObject* nilValue;
	PyObject* pyValue;
	NSObject* objValue;
}
-(instancetype)init;
-(void)dealloc;
@end

@implementation ClassWithVariables
-(instancetype)init
{
	self = [super init];
	if (self == nil) return nil;

	intValue = 42;
	floatValue = -10.055;
	charValue = 'a';
	strValue = "hello world";
	rectValue = NSMakeRect(1,2,3,4);
	nilValue = nil;
	pyValue = PySlice_New(
			PyLong_FromLong(1), 
			PyLong_FromLong(10), 
			PyLong_FromLong(4));
	objValue = [[NSObject alloc] init];
	return self;
}

-(void)dealloc
{
	PyObjC_BEGIN_WITH_GIL
		Py_XDECREF(pyValue);
	PyObjC_END_WITH_GIL
	[objValue release];
	[nilValue release];
	[super dealloc];
}

@end


static PyMethodDef mod_methods[] = {
	{ 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"instanceVariables",
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

PyObject* PyInit_instanceVariables(void);

PyObject*
PyInit_instanceVariables(void)

#else

#define INITERROR() return
#define INITDONE() return

void initinstanceVariables(void);

void
initinstanceVariables(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("instanceVariables", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}
	if (PyModule_AddObject(m, "ClassWithVariables",
		PyObjCClass_New([ClassWithVariables class])) < 0) {
		INITERROR();
	}
	INITDONE();
}
