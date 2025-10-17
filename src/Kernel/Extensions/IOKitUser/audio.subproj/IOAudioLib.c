/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
/*
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved. 
 *
 * HISTORY
 *
 */

#if 0

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/audio/IOAudioLib.h>

/* --------------------------------------------------------- */

kern_return_t
IOAudioIsOutput( io_service_t service, int *out)
{
    kern_return_t	kr;
    CFDictionaryRef	properties;
    CFNumberRef		number;

    *out = false;
    kr = IORegistryEntryCreateCFProperties(service, (CFTypeRef *) &properties,
                                        kCFAllocatorDefault, kNilOptions);
    if(kr || !properties)
        return kr;

    number = (CFNumberRef)
		CFDictionaryGetValue(properties, CFSTR("Out"));

    if( CFNumberGetTypeID() == CFGetTypeID(number))
        CFNumberGetValue(number, kCFNumberIntType, out);
    else
	kr = kIOReturnInternalError;

    CFRelease(properties);

    return kr;
}

// Tell driver when last sample will have been played, so sound hardware
// can be stopped.
kern_return_t IOAudioFlush(io_connect_t connect, IOAudioStreamPosition *end)
{
    mach_msg_type_number_t	len = 0;
    return io_connect_method_structureI_structureO(connect,
                                                   kCallFlush,
                                                   (char *)end,
                                                   sizeof(IOAudioStreamPosition),
                                                   NULL,
                                                   &len);

}

// Set autoerase flag, returns old value
kern_return_t IOAudioSetErase(io_connect_t connect, int erase, int *oldVal)
{
    kern_return_t kr;
    mach_msg_type_number_t	len = 1;
    int old;

    kr = io_connect_method_scalarI_scalarO(connect, kCallSetErase,
                &erase, 1, &old, &len);
    if(kr == kIOReturnSuccess)
	*oldVal = !old;
    return kr;
}

#endif /* 0 */

