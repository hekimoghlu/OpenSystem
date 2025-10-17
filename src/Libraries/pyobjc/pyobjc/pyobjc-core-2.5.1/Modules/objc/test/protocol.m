/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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

@protocol OC_TestProtocol
-(int)method1;
-(void)method2:(int)v;
@end

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wprotocol"
#pragma clang diagnostic ignored "-Wincomplete-implementation"
@interface OC_TestProtocolClass : NSObject <OC_TestProtocol>
{}
@end

@implementation OC_TestProtocolClass
@end
#pragma clang diagnostic pop

static PyMethodDef mod_methods[] = {
	{ 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"protocols",
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

PyObject* PyInit_protocol(void);

PyObject*
PyInit_protocol(void)

#else

#define INITERROR() return
#define INITDONE() return

void initprotocol(void);

void
initprotocol(void)
#endif
{
	PyObject* m;
	Protocol* p;


#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("protocol", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}
	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	p = @protocol(OC_TestProtocol);
	PyObject* prot = PyObjC_ObjCToPython("@", &p);
	if (!prot) {
		INITERROR();
	}
	if (PyModule_AddObject(m, "OC_TestProtocol", prot) < 0) {
		INITERROR();
	}

	INITDONE();
}
