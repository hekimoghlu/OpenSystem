/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#include "ioregpath.h"
#include <IOKit/IOKitLib.h>
#include "symbol_scope.h"

PRIVATE_EXTERN CFDictionaryRef
myIORegistryEntryCopyValue(const char * path)
{
    io_registry_entry_t 	service;
    kern_return_t       	status;
    CFMutableDictionaryRef	properties = NULL;

    service = IORegistryEntryFromPath(kIOMainPortDefault, path);
    if (service == MACH_PORT_NULL) {
	return (NULL);
    }
    status = IORegistryEntryCreateCFProperties(service,
					       &properties,
					       kCFAllocatorDefault,
					       kNilOptions);
    if (status != KERN_SUCCESS) {
	properties = NULL;
    }
    IOObjectRelease(service);
    return (properties);
}

PRIVATE_EXTERN CFTypeRef
myIORegistryEntryCopyProperty(const char * path, CFStringRef prop)
{
    io_registry_entry_t 	service;
    CFTypeRef			val;

    service = IORegistryEntryFromPath(kIOMainPortDefault, path);
    if (service == MACH_PORT_NULL) {
	return (NULL);
    }
    val = IORegistryEntryCreateCFProperty(service, prop,
					  kCFAllocatorDefault,
					  kNilOptions);
    IOObjectRelease(service);
    return (val);
}

PRIVATE_EXTERN CFDictionaryRef
myIORegistryEntryBSDNameMatchingCopyValue(const char * devname, Boolean parent)
{
    kern_return_t       	status;
    CFMutableDictionaryRef	properties = NULL;
    io_registry_entry_t 	service;

    service 
	= IOServiceGetMatchingService(kIOMainPortDefault,
				      IOBSDNameMatching(kIOMainPortDefault, 0, devname));
    if (service == MACH_PORT_NULL) {
	return (NULL);
    }
    if (parent) {
	io_registry_entry_t	parent_service;
	
	status = IORegistryEntryGetParentEntry(service, kIOServicePlane,
					       &parent_service);
	if (status == KERN_SUCCESS) {
	    status = IORegistryEntryCreateCFProperties(parent_service,
						       &properties,
						       kCFAllocatorDefault,
						       kNilOptions);
	    IOObjectRelease(parent_service);
	}
    }
    else {
	status = IORegistryEntryCreateCFProperties(service,
						   &properties,
						   kCFAllocatorDefault,
						   kNilOptions);
    }
    if (status != KERN_SUCCESS) {
	properties = NULL;
    }
    IOObjectRelease(service);
    return (properties);
}

