/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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
#ifndef __IOKIT_IOCFURLACCESS_H
#define __IOKIT_IOCFURLACCESS_H

#include <CoreFoundation/CoreFoundation.h>

CFTypeRef IOURLCreatePropertyFromResource(CFAllocatorRef alloc, CFURLRef url, CFStringRef property, SInt32 *errorCode);

Boolean IOURLCreateDataAndPropertiesFromResource(CFAllocatorRef alloc, CFURLRef url, CFDataRef *resourceData, CFDictionaryRef *properties, CFArrayRef desiredProperties, SInt32 *errorCode);

Boolean IOURLWriteDataAndPropertiesToResource(CFURLRef url, CFDataRef dataToWrite, CFDictionaryRef propertiesToWrite, int32_t *errorCode);

#ifdef HAVE_CFURLACCESS

#define kIOURLFileExists		kCFURLFileExists
#define kIOURLFileDirectoryContents	kCFURLFileDirectoryContents
#define kIOURLFileLength		kCFURLFileLength
#define kIOURLFileLastModificationTime	kCFURLFileLastModificationTime
#define kIOURLFilePOSIXMode		kCFURLFilePOSIXMode
#define kIOURLFileOwnerID		kCFURLFileOwnerID

/* Common error codes; this list is expected to grow */

typedef CFURLError IOURLError;

enum {
    kIOURLUnknownError 			= kCFURLUnknownError,
    kIOURLUnknownSchemeError 		= kCFURLUnknownSchemeError,
    kIOURLResourceNotFoundError 	= kCFURLResourceNotFoundError,
    kIOURLResourceAccessViolationError 	= kCFURLResourceAccessViolationError,
    kIOURLRemoteHostUnavailableError 	= kCFURLRemoteHostUnavailableError,
    kIOURLImproperArgumentsError 	= kCFURLImproperArgumentsError,
    kIOURLUnknownPropertyKeyError 	= kCFURLUnknownPropertyKeyError,
    kIOURLPropertyKeyUnavailableError 	= kCFURLPropertyKeyUnavailableError,
    kIOURLTimeoutError 			= kCFURLTimeoutError
};

#else /* !HAVE_CFURLACCESS */

#define kIOURLFileExists		CFSTR("kIOURLFileExists")
#define kIOURLFileDirectoryContents	CFSTR("kIOURLFileDirectoryContents")
#define kIOURLFileLength		CFSTR("kIOURLFileLength")
#define kIOURLFileLastModificationTime	CFSTR("kIOURLFileLastModificationTime")
#define kIOURLFilePOSIXMode		CFSTR("kIOURLFilePOSIXMode")
#define kIOURLFileOwnerID		CFSTR("kIOURLFileOwnerID")

/* Common error codes; this list is expected to grow */

typedef enum {
      kIOURLUnknownError = -10,
          kIOURLUnknownSchemeError = -11,
          kIOURLResourceNotFoundError = -12,
          kIOURLResourceAccessViolationError = -13,
          kIOURLRemoteHostUnavailableError = -14,
          kIOURLImproperArgumentsError = -15,
          kIOURLUnknownPropertyKeyError = -16,
          kIOURLPropertyKeyUnavailableError = -17,
          kIOURLTimeoutError = -18
} IOURLError;

#endif /* !HAVE_CFURLACCESS */

#endif /* __IOKIT_IOCFURLACCESS_H */
