/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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
#include <stdarg.h>

#import <Foundation/Foundation.h>

typedef struct _Foo* FooHandle;
typedef struct _Bar* BarHandle;

@interface NSObject (OC_LockingTest)
-(void)setLocked:(NSObject*)value;
-(NSObject*)isLocked;
-(void)appendToList:(NSObject*)value;
@end

@interface OC_LockTest : NSObject
-(void)threadFunc:(NSObject*)object;
@end

@implementation OC_LockTest
-(void)threadFunc:(NSObject*)object
{
	int i;
	for (i = 0; i < 6; i++) {
		usleep(500000);
		@synchronized(object) {
			NSNumber* isLocked = (NSNumber*)[object isLocked];
			if ([isLocked boolValue]) {
				[object appendToList:@"LOCK FOUND"];
			}
			[object setLocked:[NSNumber numberWithBool:YES]];
			[object appendToList:@"threading a"];
			usleep(5000000);
			[object appendToList:@"threading b"];
			[object setLocked:[NSNumber numberWithBool:NO]];
		}
	}
}
@end


static PyMethodDef mod_methods[] = {
	        { 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"locking",
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

PyObject* PyInit_locking(void);

PyObject*
PyInit_locking(void)

#else

#define INITERROR() return
#define INITDONE() return

void initlocking(void);

void
initlocking(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("locking", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	if (PyModule_AddObject(m, "OC_LockTest", 
		PyObjCClass_New([OC_LockTest class])) < 0) {
		INITERROR();
	}

	INITDONE();
}
