/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 7, 2025.
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

@interface OC_TestFilePointer : NSObject
{
}

-(FILE*)openFile:(char*)path withMode:(char*)mode;
-(NSString*)readline:(FILE*)fp;
@end

@implementation OC_TestFilePointer
-(FILE*)openFile:(char*)path withMode:(char*)mode
{
	return fopen(path, mode);
}

-(NSString*)readline:(FILE*)fp
{
	char buf[1024];

	return [NSString stringWithCString: fgets(buf, sizeof(buf), fp)
		         encoding:NSASCIIStringEncoding];
}
@end

static PyMethodDef mod_methods[] = {
	{ 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"filepointer",
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

PyObject* PyInit_filepointer(void);

PyObject*
PyInit_filepointer(void)

#else

#define INITERROR() return
#define INITDONE() return

void initfilepointer(void);

void
initfilepointer(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("filepointer", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	if (PyModule_AddObject(m, "OC_TestFilePointer", 
			PyObjCClass_New([OC_TestFilePointer class])) < 0) {
		INITERROR();
	}

	INITDONE();
}
