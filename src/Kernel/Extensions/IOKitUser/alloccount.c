/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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
cc alloccount.c -o alloccount -Wall -framework IOKit
 */

#include <assert.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitKeys.h>

static void printNumber( CFDictionaryRef dict, CFStringRef name )
{
    CFNumberRef	num;
    SInt32	num32;

    num = (CFNumberRef) CFDictionaryGetValue(dict, name);
    if( num) {
        assert( CFNumberGetTypeID() == CFGetTypeID(num) );
        CFNumberGetValue(num, kCFNumberSInt32Type, &num32);
	printf("%22s = %08lx = %ld K\n",
                CFStringGetCStringPtr(name, CFStringGetSystemEncoding()),
                num32, num32 / 1024);
    }
}

int main(int argc, char **argv)
{
    mach_port_t		   masterPort;
    io_registry_entry_t	   root;
    CFDictionaryRef        props;
    kern_return_t          status;

    // Obtain the I/O Kit communication handle.

    status = IOMasterPort(bootstrap_port, &masterPort);
    assert(status == KERN_SUCCESS);

    // Obtain the registry root entry.

    root = IORegistryGetRootEntry(masterPort);
    assert(root);

    status = IORegistryEntryCreateCFProperties(root,
			(CFTypeRef *) &props,
			kCFAllocatorDefault, kNilOptions );
    assert( KERN_SUCCESS == status );
    assert( CFDictionaryGetTypeID() == CFGetTypeID(props));

    props = (CFDictionaryRef)
		CFDictionaryGetValue( props, CFSTR(kIOKitDiagnosticsKey));
    assert( props );
    assert( CFDictionaryGetTypeID() == CFGetTypeID(props));

    printNumber(props, CFSTR("Instance allocation"));
    printNumber(props, CFSTR("Container allocation"));
    printNumber(props, CFSTR("IOMalloc allocation"));

    CFRelease(props);
    IOObjectRelease(root);

    exit(0);	
}

