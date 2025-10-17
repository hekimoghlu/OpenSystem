/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
 *  BLGetRAIDBootData.c
 *  bless
 *
 *  Created by Shantonu Sen on 1/13/05.
 *  Copyright 2005-2007 Apple Inc. All Rights Reserved.
 *
 *
 *  $Id: BLGetRAIDBootDataForDevice.c,v 1.8 2006/02/20 22:49:58 ssen Exp $
 *
 */

#include <stdlib.h>
#include <unistd.h>

#include <mach/mach_error.h>

#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitKeys.h>

#include <CoreFoundation/CoreFoundation.h>

#include "bless.h"
#include "bless_private.h"

#if SUPPORT_RAID

#include <IOKit/storage/RAID/AppleRAIDUserLib.h>

int BLGetRAIDBootDataForDevice(BLContextPtr context, const char * device,
							   CFTypeRef *bootData)
{
	const char *name = NULL;
	kern_return_t           kret;
	mach_port_t             ourIOKitPort;
	io_service_t			service;
	io_iterator_t			serviter;
	CFTypeRef			data = NULL;
	
	*bootData = NULL;
	
	if(!device || 0 != strncmp(device, "/dev/", 5)) return 1;

	name = device + 5;
	
	// Obtain the I/O Kit communication handle.
        if((kret = IOMasterPort(bootstrap_port, &ourIOKitPort)) != KERN_SUCCESS) {
            return 2;
        }	
	
	kret =  IOServiceGetMatchingServices(ourIOKitPort,
                                             IOBSDNameMatching(ourIOKitPort,
                                                               0, name),
                                             &serviter);
	if (kret != KERN_SUCCESS) {
            return 3;
        }
	
        service = IOIteratorNext(serviter);
        if (!service) {
            IOObjectRelease(serviter);
            return 3;
        }
        
	IOObjectRelease(serviter);
	
	
	// we know this IOService is a RAID member. Now we need to get the boot data
	data = IORegistryEntrySearchCFProperty( service,
											  kIOServicePlane,
											  CFSTR(kIOBootDeviceKey),
											  kCFAllocatorDefault,
											  kIORegistryIterateRecursively|
											  kIORegistryIterateParents);
	if(data == NULL) {
		// it's an error for a RAID not to have this information
		IOObjectRelease(service);
		return 0;
	}
	
	IOObjectRelease(service);
	
	if(CFGetTypeID(data) == CFArrayGetTypeID()) {

	} else if(CFGetTypeID(bootData) == CFDictionaryGetTypeID()) {
		
	} else {
		contextprintf(context, kBLLogLevelError,  "Invalid RAID boot data\n" );
		return 3;                
	}
	
	*bootData = data;
	
	return 0;
}

#endif // SUPPORT_RAID
