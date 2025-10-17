/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
 *  BLSupportsLegacyMode.c
 *  bless
 *
 *  Created by Shantonu Sen on 2/10/06.
 *  Copyright 2006 Apple Computer. All Rights Reserved.
 *
 */

#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitKeys.h>

#include <sys/stat.h>
#include "bless.h"
#include "bless_private.h"

// Check if a system supports CSM legacy mode

#define kBL_APPLE_VENDOR_NVRAM_GUID "4D1EDE05-38C7-4A6A-9CC6-4BCCA8B38C14"
static bool _getFeatureFlags(BLContextPtr context, uint32_t *featureMask,
                        uint32_t *featureFlags);

bool BLSupportsLegacyMode(BLContextPtr context)
{
    
	uint32_t		featureMask;
	uint32_t		featureFlags;
    
    if(!_getFeatureFlags(context, &featureMask, &featureFlags)) {
        return false;
    }

	if((featureMask & 0x00000001)
	   && (featureFlags & 0x00000001)) {
        contextprintf(context, kBLLogLevelVerbose,  "Legacy mode suppported\n");
		return true;
	}
	
    contextprintf(context, kBLLogLevelVerbose,  "Legacy mode NOT suppported\n");
    return false;
}

static bool _getFeatureFlags(BLContextPtr context, uint32_t *featureMask,
                        uint32_t *featureFlags)
{
    
    io_registry_entry_t optionsNode = IO_OBJECT_NULL;
    CFDataRef		dataRef;
    
    optionsNode = IORegistryEntryFromPath(kIOMasterPortDefault, kIODeviceTreePlane ":/options");
    
    if(IO_OBJECT_NULL == optionsNode) {
        contextprintf(context, kBLLogLevelVerbose,  "Could not find " kIODeviceTreePlane ":/options\n");
        return false;
    }
    
    dataRef = IORegistryEntryCreateCFProperty(optionsNode,
											 CFSTR(kBL_APPLE_VENDOR_NVRAM_GUID ":FirmwareFeaturesMask"),
											 kCFAllocatorDefault, 0);

	if(dataRef != NULL
	   && CFGetTypeID(dataRef) == CFDataGetTypeID()
	   && CFDataGetLength(dataRef) == sizeof(uint32_t)) {
		const UInt8	*bytes = CFDataGetBytePtr(dataRef);
		
		*featureMask = CFSwapInt32LittleToHost(*(uint32_t *)bytes);
	} else {
		*featureMask = 0x000003FF;
	}
    
	if(dataRef) CFRelease(dataRef);
	
    dataRef = IORegistryEntryCreateCFProperty(optionsNode,
											  CFSTR(kBL_APPLE_VENDOR_NVRAM_GUID ":FirmwareFeatures"),
											  kCFAllocatorDefault, 0);
	
	if(dataRef != NULL
	   && CFGetTypeID(dataRef) == CFDataGetTypeID()
	   && CFDataGetLength(dataRef) == sizeof(uint32_t)) {
		const UInt8	*bytes = CFDataGetBytePtr(dataRef);
		
		*featureFlags = CFSwapInt32LittleToHost(*(uint32_t *)bytes);
	} else {
		*featureFlags = 0x00000014;
	}
    
	if(dataRef) CFRelease(dataRef);
		
	IOObjectRelease(optionsNode);
    
	contextprintf(context, kBLLogLevelVerbose,  "Firmware feature mask: 0x%08X\n", *featureMask);
	contextprintf(context, kBLLogLevelVerbose,  "Firmware features: 0x%08X\n", *featureFlags);
	
    return true;
}
