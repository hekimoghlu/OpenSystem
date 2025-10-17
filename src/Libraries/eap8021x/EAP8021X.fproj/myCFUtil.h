/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#ifndef _S_MYCFUTIL_H
#define _S_MYCFUTIL_H

/* 
 * Modification History
 *
 * November 8, 2001	Dieter Siegmund
 * - created
 */

#include <CoreFoundation/CFString.h>
#include <mach/mach.h>

Boolean
my_CFEqual(CFTypeRef val1, CFTypeRef val2);

void 
my_CFRelease(void * t);

char *
my_CFStringToCString(CFStringRef cfstr, CFStringEncoding encoding);

CFStringRef
my_CFStringCreateWithCString(const char * cstr);

CFPropertyListRef 
my_CFPropertyListCreateFromFile(const char * filename);

int
my_CFPropertyListWriteFile(CFPropertyListRef plist, const char * filename);

Boolean
my_CFDictionaryGetBooleanValue(CFDictionaryRef properties, CFStringRef propname,
			       Boolean def_value);

CFPropertyListRef
my_CFPropertyListCreateWithBytePtrAndLength(const void * data, int data_len);

CFStringRef
my_CFUUIDStringCreate(CFAllocatorRef alloc);

CFStringRef
my_CFStringCreateWithData(CFDataRef data);

CFDataRef
my_CFDataCreateWithString(CFStringRef str);

void
my_FieldSetRetainedCFType(void * field_p, const void * v);

CFStringRef
my_CFPropertyListCopyAsXMLString(CFPropertyListRef plist);

#define STRING_APPEND(__string, __format, ...)		\
    CFStringAppendFormat(__string, NULL,		\
			 CFSTR(__format),		\
			 ## __VA_ARGS__)

vm_address_t
my_CFPropertyListCreateVMData(CFPropertyListRef plist,
			      mach_msg_type_number_t * 	ret_data_len);

CFStringRef
my_CFStringCopyComponent(CFStringRef path, CFStringRef separator, 
			 CFIndex component_index);
#endif /* _S_MYCFUTIL_H */
