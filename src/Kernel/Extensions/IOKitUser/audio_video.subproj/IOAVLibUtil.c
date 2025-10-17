/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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
#include "IOAVLibUtil.h"

#include <AssertMacros.h>


static bool RegistryPathStringContainsPlane(CFStringRef path)
{
    CFRange range = CFStringFind(path, CFSTR(":"), 0);

    return ( range.location != kCFNotFound ) && ( ( range.location + range.length ) < CFStringGetLength(path) );
}

static CFStringRef CreateAbsoluteRegistryPathFromString(CFStringRef path, const io_name_t defaultPlane)
{
    if ( ! RegistryPathStringContainsPlane(path) )
        path = CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("%s:%@"), defaultPlane, path);
    else
        CFRetain(path);

    return path;
}

CFMutableDictionaryRef __IOAVClassMatching(const char * typeName, CFStringRef rootPath, IOAVLocation location, unsigned int unit)
{
#if defined(__arm__) || defined(__arm64__)
    CFMutableDictionaryRef  matching    = NULL;

    matching = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    require(matching, exit);

    // filter by location, if specified
    if ( location < kIOAVLocationCount ) {
        CFStringRef locString   = NULL;

        locString = CFStringCreateWithCString(kCFAllocatorDefault, IOAVLocationString(location), kCFStringEncodingUTF8);
        require(locString, error);

        CFDictionarySetValue(matching, CFSTR(kIOAVLocationKey), locString);

        CFRelease(locString);
    }

    // filter by unit, if specified
    if ( unit != kIOAVUnitNone ) {
        CFNumberRef num = NULL;

        num = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &unit);
        require(num, error);

        CFDictionarySetValue(matching, CFSTR(kIOAVUnitKey), num);

        CFRelease(num);
    }

    // filter by root path, if specified
    if ( rootPath ) {
        CFMutableDictionaryRef parentMatching;
        CFStringRef path;

        path = CreateAbsoluteRegistryPathFromString(rootPath, kIODeviceTreePlane);
        require(path, error);

        parentMatching = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        require_action(parentMatching, error, CFRelease(path));

        CFDictionarySetValue(parentMatching, CFSTR(kIOPathMatchKey), path);
        CFRelease(path);

        CFDictionarySetValue(matching, CFSTR(kIOParentMatchKey), parentMatching);

        CFRelease(parentMatching);
    }

    // Append property to match
    {
        CFMutableDictionaryRef propertyDict = NULL;
        CFStringRef matchProp = NULL;

        propertyDict = CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        require(propertyDict, error);

        matchProp = CFStringCreateWithFormat(kCFAllocatorDefault, NULL, CFSTR("%s%s"), typeName, kIOAVUserInterfaceSupportedKeySuffix);
        require_action(matchProp, error, CFRelease(propertyDict));

        CFDictionarySetValue(propertyDict, matchProp, kCFBooleanTrue);

        CFDictionarySetValue(matching, CFSTR(kIOPropertyMatchKey), propertyDict);

        CFRelease(propertyDict);
        CFRelease(matchProp);
    }

    goto exit;

error:
    CFRelease(matching);
    matching = NULL;

exit:
    return matching;
#else
    (void)typeName;
    (void)rootPath;
    (void)location;
    (void)unit;
    (void)CreateAbsoluteRegistryPathFromString;
    return NULL;
#endif // defined(__arm__) || defined(__arm64__)
}

CFTypeRef __IOAVCopyFirstMatchingIOAVObjectOfType(const char * typeName, IOAVTypeConstructor * typeConstructor, CFStringRef rootPath, IOAVLocation location, unsigned int unit)
{
    CFTypeRef               object      = NULL;
    io_service_t            service;

    service = IOServiceGetMatchingService(kIOMasterPortDefault, IOAVClassMatching(typeName, rootPath, location, unit));
    require(service, exit);

    object = typeConstructor(kCFAllocatorDefault, service);

exit:
    if ( service )
        IOObjectRelease(service);

    return object;
}
