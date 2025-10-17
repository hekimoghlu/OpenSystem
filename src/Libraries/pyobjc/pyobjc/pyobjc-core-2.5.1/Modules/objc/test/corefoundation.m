/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#import <CoreFoundation/CoreFoundation.h>


@interface OC_TestCoreFoundation : NSObject
{
}
// not toll-free bridged. 
+(char*)signatureForCFUUIDRef;
+(CFTypeID)typeidForCFUUIDRef;
+(CFUUIDRef)createUUID;
+(NSString*)formatUUID:(CFUUIDRef)uuid;
+(NSObject*)anotherUUID;

// tollfree bridged:
+(char*)signatureForCFDateRef;
+(CFTypeID)typeidForCFDateRef;
+(CFDateRef)today;
+(NSString*)formatDate:(CFDateRef)date;
+(int)shortStyle;
@end


@implementation OC_TestCoreFoundation

+(char*)signatureForCFUUIDRef
{
	return @encode(CFUUIDRef);
}

+(CFTypeID)typeidForCFUUIDRef
{
	return CFUUIDGetTypeID();
}

+(CFUUIDRef)createUUID
{
	CFUUIDRef result =  CFUUIDCreate(NULL);

	/* We own a reference, but want to released a borrowed ref. */
	[(NSObject*)result retain];
	CFRelease(result);
	[(NSObject*)result autorelease];

	return result;
}

+(NSObject*)anotherUUID
{
	CFUUIDRef result =  CFUUIDCreate(NULL);

	/* We own a reference, but want to released a borrowed ref. */
	[(NSObject*)result autorelease];

	return (NSObject*)result;
}


+(NSString*)formatUUID:(CFUUIDRef)uuid
{
	NSString* result;

	result = (NSString*)CFUUIDCreateString(NULL, uuid);
	return [result autorelease];
}



+(char*)signatureForCFDateRef
{
	return @encode(CFDateRef);
}

+(CFTypeID)typeidForCFDateRef
{
	return CFDateGetTypeID();
}

+(CFDateRef)today
{
	CFDateRef result;

	result = CFDateCreate(NULL, CFAbsoluteTimeGetCurrent());

	/* We own a reference, but want to released a borrowed ref. */
	[(NSObject*)result autorelease];

	return result;
}

+(NSString*)formatDate:(CFDateRef)date
{
	CFLocaleRef currentLocale = CFLocaleCopyCurrent();
	CFDateFormatterRef formatter = CFDateFormatterCreate(
			NULL, currentLocale, 
			kCFDateFormatterShortStyle, kCFDateFormatterNoStyle  );

	if (currentLocale != NULL) {
		CFRelease(currentLocale);
	}

	NSString* result = (NSString*)CFDateFormatterCreateStringWithDate(
			NULL, formatter, date);

	CFRelease(formatter);
	return [result autorelease];
}

+(int)shortStyle
{
	return kCFDateFormatterShortStyle;
}

@end



static PyMethodDef mod_methods[] = {
	        { 0, 0, 0, 0 }
};
#if PY_VERSION_HEX >= 0x03000000

static struct PyModuleDef mod_module = {
	PyModuleDef_HEAD_INIT,
	"corefoundation",
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

PyObject* PyInit_corefoundation(void);

PyObject*
PyInit_corefoundation(void)

#else

#define INITERROR() return
#define INITDONE() return

void initcorefoundation(void);

void
initcorefoundation(void)
#endif
{
	PyObject* m;

#if PY_VERSION_HEX >= 0x03000000
	m = PyModule_Create(&mod_module);
#else
	m = Py_InitModule4("corefoundation", mod_methods,
		NULL, NULL, PYTHON_API_VERSION);
#endif
	if (!m) {
		INITERROR();
	}

	if (PyObjC_ImportAPI(m) < 0) {
		INITERROR();
	}

	if (PyModule_AddObject(m, "OC_TestCoreFoundation", 
		PyObjCClass_New([OC_TestCoreFoundation class])) < 0) {
		INITERROR();
	}

	INITDONE();
}
