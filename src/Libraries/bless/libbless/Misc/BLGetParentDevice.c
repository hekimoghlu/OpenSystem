/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
 *  BLGetParentDevice.c
 *  bless
 *
 *  Created by Shantonu Sen <ssen@apple.com> on Mon Jun 25 2001.
 *  Copyright (c) 2001-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLGetParentDevice.c,v 1.19 2006/02/20 22:49:56 ssen Exp $
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/param.h>
#include <sys/mount.h>

#import <mach/mach_error.h>
#import <APFS/APFS.h>
#import <IOKit/IOKitLib.h>
#import <IOKit/IOBSD.h>
#import <IOKit/storage/IOMedia.h>
#import <IOKit/storage/IOPartitionScheme.h>

#include <CoreFoundation/CoreFoundation.h>

#include "bless.h"
#include "bless_private.h"

int BLGetParentDevice(BLContextPtr context,  const char * partitionDev,
		      char * parentDev,
		      uint32_t parentDevSize,
		      uint32_t *partitionNum) {

    return BLGetParentDeviceAndPartitionType(context, partitionDev, parentDev, parentDevSize, partitionNum, NULL);
}

    
int BLGetParentDeviceAndPartitionType(BLContextPtr context,   const char * partitionDev,
			 char * parentDev,
			 uint32_t parentDevSize,
			 uint32_t *partitionNum,
			BLPartitionType *partitionType) {

    int                     result = 0;
    kern_return_t           kret;
    io_iterator_t           services = MACH_PORT_NULL;
    io_iterator_t           parents = MACH_PORT_NULL;
    io_registry_entry_t     service = MACH_PORT_NULL;
    io_iterator_t           grandparents = MACH_PORT_NULL;
    io_registry_entry_t     service2 = MACH_PORT_NULL;
    io_object_t             obj = MACH_PORT_NULL;
    CFNumberRef             pn = NULL;
    CFStringRef             content = NULL;

    char par[MNAMELEN];

    parentDev[0] = '\0';

    kret = IOServiceGetMatchingServices(kIOMasterPortDefault,
					IOBSDNameMatching(kIOMasterPortDefault,
							  0,
							  (char *)partitionDev + 5),
					&services);
    if (kret != KERN_SUCCESS) {
      result = 3;
      goto finish;
    }

    // Should only be one IOKit object for this volume. (And we only want one.)
    obj = IOIteratorNext(services);
    if (!obj) {
        result = 4;
        goto finish;
    }  

    // we have the IOMedia for the partition.

    if (IOObjectConformsTo(obj, APFS_VOLUME_OBJECT)) {
        result = 10;
        goto finish;
    }
    
    pn = (CFNumberRef)IORegistryEntryCreateCFProperty(obj, CFSTR(kIOMediaPartitionIDKey),
                                                      kCFAllocatorDefault, 0);
    
    if (pn == NULL) {
        result = 4;
        goto finish;
    }
    
    if (CFGetTypeID(pn) != CFNumberGetTypeID()) {
        result = 5;
        goto finish;
    }
    
    CFNumberGetValue(pn, kCFNumberSInt32Type, partitionNum);
    
    kret = IORegistryEntryGetParentIterator (obj, kIOServicePlane,
					       &parents);
    if (kret) {
      result = 6;
      goto finish;
      /* We'll never loop forever. */
    }

    while ( (service = IOIteratorNext(parents)) != 0 ) {

        kret = IORegistryEntryGetParentIterator (service, kIOServicePlane,
                                                &grandparents);
        IOObjectRelease(service);
        service = MACH_PORT_NULL;

        if (kret) {
            result = 6;
            goto finish;
            /* We'll never loop forever. */
        }

        while ( (service2 = IOIteratorNext(grandparents)) != 0 ) {
        
            if (content) {
                CFRelease(content);
                content = NULL;
            }

            if (!IOObjectConformsTo(service2, "IOMedia")) {
                IOObjectRelease(service2);
                service2 = MACH_PORT_NULL;
                continue;
            }
        
            content = (CFStringRef)
                IORegistryEntryCreateCFProperty(service2,
                                                CFSTR(kIOMediaContentKey),
                                                kCFAllocatorDefault, 0);
            
            
            if(CFGetTypeID(content) != CFStringGetTypeID()) {
                result = 2;
                goto finish;
            }
            
            if(CFStringCompare(content, CFSTR("Apple_partition_scheme"), 0)
               == kCFCompareEqualTo) {
                if(partitionType) *partitionType = kBLPartitionType_APM;
            } else if(CFStringCompare(content, CFSTR("FDisk_partition_scheme"), 0)
                      == kCFCompareEqualTo) {
                if(partitionType) *partitionType = kBLPartitionType_MBR;
            } else if(CFStringCompare(content, CFSTR("GUID_partition_scheme"), 0)
                      == kCFCompareEqualTo) {
                if(partitionType) *partitionType = kBLPartitionType_GPT;
            } else {
                IOObjectRelease(service2);
                service2 = MACH_PORT_NULL;
                CFRelease(content);
                content = NULL;
                continue;
            }

            CFRelease(content);

            content = IORegistryEntryCreateCFProperty(service2, CFSTR(kIOBSDNameKey),
                                                        kCFAllocatorDefault, 0);
        
            if(CFGetTypeID(content) != CFStringGetTypeID()) {
                result = 3;
                goto finish;
            }
        
            if(!CFStringGetCString(content, par, MNAMELEN, kCFStringEncodingASCII)) {
                result = 4;
                goto finish;
            }

            CFRelease(content);
            content = NULL;

            snprintf(parentDev, parentDevSize, "/dev/%s",par);
            break;
        }

        if(parentDev[0] == '\0') {
            break;
        }
    }

    if(parentDev[0] == '\0') {
      // nothing found
      result = 8;
      goto finish;
    }

finish:
    if (services != MACH_PORT_NULL)     IOObjectRelease(services);
    if (parents != MACH_PORT_NULL)      IOObjectRelease(parents);
    if (service != MACH_PORT_NULL)      IOObjectRelease(service);
    if (grandparents != MACH_PORT_NULL) IOObjectRelease(grandparents);
    if (service2 != MACH_PORT_NULL)     IOObjectRelease(service2);
    if (obj != MACH_PORT_NULL)          IOObjectRelease(obj);
    if (pn)                             CFRelease(pn);
    if (content)                        CFRelease(content);
    
    return result;
}
