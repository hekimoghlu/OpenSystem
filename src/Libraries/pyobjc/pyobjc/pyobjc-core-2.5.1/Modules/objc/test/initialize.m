/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

static int numUninitialized = 0;

@interface OC_TestInitialize : NSObject
{
	int       isInitialized;
}
-(instancetype)init;
-(instancetype)retain;
-(void)release;
-(instancetype)autorelease;
-(int)isInitialized;
+(int)numUninitialized;
-(id)dummy;
+(id)makeInstance;

/* completely unrelated ... */
-(oneway void)onewayVoidMethod;

@end

@implementation OC_TestInitialize 

-(instancetype)init
{
	self = [super init];
	if (!self) return self;

	isInitialized = 1;
	return self;
}

-(instancetype)retain
{
	if (!isInitialized) {
		numUninitialized ++;
	}
	return [super retain];
}

-(void)release
{
	if (!isInitialized) {
		numUninitialized ++;
	}
	[super release];
}

-(instancetype)autorelease
{
	if (!isInitialized) {
		numUninitialized ++;
	}
	return [super autorelease];
}

-(int)isInitialized
{
	return isInitialized;
}

+(int)numUninitialized
{
	return numUninitialized;
}

-(id)dummy
{
	return @"hello";
}

+(id)makeInstance
{
	return [[[self alloc] init] autorelease];
}

-(oneway void)onewayVoidMethod
{
	isInitialized=-1;
}

@end


static PyMethodDef mod_methods[] = {
	{ 0, 0, 0, 0 }
};

#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"initialize",
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

PyObject* PyInit_initialize(void);

PyObject*
PyInit_initialize(void)

#else

#define INITERROR() return
#define INITDONE() return

void initinitialize(void);

void
initinitialize(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("initialize", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}
	if (PyModule_AddObject(m, "OC_TestInitialize", 
		PyObjCClass_New([OC_TestInitialize class])) < 0) {
		INITERROR();
	}

	INITDONE();
}
