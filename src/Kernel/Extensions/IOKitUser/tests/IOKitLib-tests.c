/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#include <darwintest.h>

#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>

T_DECL(IOMasterPort,
       "check if one can retrieve mach port for communicating with IOKit",
       T_META_NAMESPACE("IOKitUser.IOKitLib")
       )
{
    mach_port_t masterPort = MACH_PORT_NULL;

    T_EXPECT_MACH_SUCCESS(IOMasterPort(MACH_PORT_NULL, &masterPort), NULL);
    T_EXPECT_NE(MACH_PORT_NULL, masterPort, NULL);
}

T_DECL(OSNumberFloats,
       "check roundtrip serialization of float/double CFNumber serialization",
       T_META_NAMESPACE("IOKitUser.IOKitLib")
       )
{
	CFMutableDictionaryRef dict, props;
	CFNumberRef num;

	dict = CFDictionaryCreateMutable( kCFAllocatorDefault, 0,
			&kCFTypeDictionaryKeyCallBacks,
			&kCFTypeDictionaryValueCallBacks);
	props = CFDictionaryCreateMutable( kCFAllocatorDefault, 0,
			&kCFTypeDictionaryKeyCallBacks,
			&kCFTypeDictionaryValueCallBacks);
	CFDictionarySetValue(props, CFSTR("OSNumberFloatTest"), dict);

	float floatValue = 1234.5678;
    num = CFNumberCreate( kCFAllocatorDefault, kCFNumberFloatType, &floatValue );
	CFDictionarySetValue(dict, CFSTR("floatValue"), num);
	CFRelease(num);

	double doubleValue = 5678.1234;
    num = CFNumberCreate( kCFAllocatorDefault, kCFNumberDoubleType, &doubleValue );
	CFDictionarySetValue(dict, CFSTR("doubleValue"), num);
	CFRelease(num);

	SInt64 intValue = 12345678;
    num = CFNumberCreate( kCFAllocatorDefault, kCFNumberSInt64Type, &intValue );
	CFDictionarySetValue(dict, CFSTR("intValue"), num);
	CFRelease(num);

	kern_return_t kr;
	io_service_t
	service = IORegistryEntryFromPath(kIOMasterPortDefault, kIOServicePlane ":/IOResources");
    T_EXPECT_NE(MACH_PORT_NULL, service, NULL);

	kr = IORegistryEntrySetCFProperties(service, props);
    T_EXPECT_MACH_SUCCESS(kr, NULL);

	CFTypeRef obj = IORegistryEntryCreateCFProperty(service, CFSTR("OSNumberFloatTest"), kCFAllocatorDefault, 0);
    T_EXPECT_NE(NULL, obj, NULL);
    T_EXPECT_TRUE(CFEqual(obj, dict), NULL);

	CFRelease(obj);
	CFRelease(dict);
	CFRelease(props);
}
