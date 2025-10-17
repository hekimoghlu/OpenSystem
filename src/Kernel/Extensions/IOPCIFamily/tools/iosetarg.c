/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <IOKit/IOKitLib.h>
#include <mach/mach_error.h>

#ifndef kIODebugArgumentsKey
#define kIODebugArgumentsKey "IODebugArguments"
#endif

int main(int argc, char * argv[])
{
	kern_return_t          kr;
    io_service_t           service;
    CFStringRef            str;
    CFMutableArrayRef	   array;
    CFMutableDictionaryRef matching;
    uint32_t	           idx;
    uint64_t               id;

	if (argc < 3) exit(1);

    id = strtoll(argv[1], NULL, 0);

    matching = id ? IORegistryEntryIDMatching(id) : IOServiceMatching(argv[1]);

	array = CFArrayCreateMutable(kCFAllocatorDefault, argc - 2, &kCFTypeArrayCallBacks);

	for (idx = 2; idx < argc; idx++)
	{
		str = CFStringCreateWithCString(kCFAllocatorDefault, argv[idx], CFStringGetSystemEncoding());
		CFArrayAppendValue(array, str);
	}

    service = IOServiceGetMatchingService(kIOMainPortDefault, matching);
    assert(service);
	kr = IORegistryEntrySetCFProperty(service, CFSTR(kIODebugArgumentsKey), array);

	printf("result: 0x%x, %s\n", kr, mach_error_string(kr));

	exit(0);
}
