/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
cc classcount.c -o classcount -Wall -framework IOKit
 */

#include <assert.h>

#include <CoreFoundation/CoreFoundation.h>

#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitKeys.h>

int main(int argc, char **argv)
{
    kern_return_t	status;
    mach_port_t		masterPort;
    io_registry_entry_t	root;
    CFDictionaryRef	dictionary;
    CFDictionaryRef	props;
    CFStringRef		key;
    CFNumberRef		num;
    int			arg;

    // Parse args

    if( argc < 2 ) {
	printf("%s ClassName...\n", argv[0]);
	exit(0);
    }

    // Obtain the I/O Kit communication handle.

    status = IOMasterPort(bootstrap_port, &masterPort);
    assert(status == KERN_SUCCESS);

    // Obtain the registry root entry.

    root = IORegistryGetRootEntry(masterPort);
    assert(root);

    status = IORegistryEntryCreateCFProperties(root,
			(CFTypeRef *)&props,
			kCFAllocatorDefault, kNilOptions );
    assert( KERN_SUCCESS == status );
    assert( CFDictionaryGetTypeID() == CFGetTypeID(props));

    dictionary = (CFDictionaryRef)
		CFDictionaryGetValue( props, CFSTR(kIOKitDiagnosticsKey));
    assert( dictionary );
    assert( CFDictionaryGetTypeID() == CFGetTypeID(dictionary));

    dictionary = (CFDictionaryRef)
		CFDictionaryGetValue( dictionary, CFSTR("Classes"));
    assert( dictionary );
    assert( CFDictionaryGetTypeID() == CFGetTypeID(dictionary));

    for( arg = 1; arg < argc; arg++ ) {
	key = CFStringCreateWithCString(kCFAllocatorDefault,
			argv[arg], CFStringGetSystemEncoding());
	assert(key);
        num = (CFNumberRef) CFDictionaryGetValue(dictionary, key);
	CFRelease(key);
        if( num) {
	    SInt32	num32;
            assert( CFNumberGetTypeID() == CFGetTypeID(num) );
	    CFNumberGetValue(num, kCFNumberSInt32Type, &num32);
            printf("%s = %d, ", argv[arg], (int)num32);
	}
    }
    if( num)
        printf("\n");

    CFRelease(props);
    IOObjectRelease(root);

    exit(0);	
}

