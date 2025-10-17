/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
 *  BLGetOpenFirmwareBootDeviceForNetworkPath.c
 *  bless
 *
 *  Created by Shantonu Sen on 4/11/06.
 *  Copyright 2006-2007 Apple Inc. All Rights Reserved.
 *
 * $Id: BLGetOpenFirmwareBootDeviceForNetworkPath.c,v 1.1 2006/04/12 00:15:05 ssen Exp $
 *
 */

#import <IOKit/IOKitLib.h>
#import <IOKit/IOBSD.h>
#import <IOKit/IOKitKeys.h>
#include <IOKit/network/IONetworkInterface.h>
#include <IOKit/network/IONetworkController.h>

#import <CoreFoundation/CoreFoundation.h>

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/param.h>

#include "bless.h"
#include "bless_private.h"

int BLGetOpenFirmwareBootDeviceForNetworkPath(BLContextPtr context,
                                               const char *interface,
                                               const char *host,
                                               const char *path,
											   char * ofstring,
											   uint32_t ofstringSize) {

    mach_port_t masterPort;
    kern_return_t kret;
    io_service_t iface, service;
	io_iterator_t iter;
	io_string_t	pathInPlane;
	bool gotPathInPlane = false;

    CFMutableDictionaryRef matchDict;

    kret = IOMasterPort(MACH_PORT_NULL, &masterPort);
    if(kret) return 1;
        
    
    matchDict = IOBSDNameMatching(masterPort, 0, interface);
    CFDictionarySetValue(matchDict, CFSTR(kIOProviderClassKey), CFSTR(kIONetworkInterfaceClass));


    iface = IOServiceGetMatchingService(masterPort,
                                        matchDict);
    
    if(iface == IO_OBJECT_NULL) {
        contextprintf(context, kBLLogLevelError, "Could not find object for %s\n", interface);
        return 1;
    }
	
	// find this the parent that's in the device tree plane
	kret = IORegistryEntryCreateIterator(iface, kIOServicePlane,
		kIORegistryIterateRecursively|kIORegistryIterateParents,
		&iter);
	IOObjectRelease(iface);
	
	if(kret) {
        contextprintf(context, kBLLogLevelError, "Could not find object for %s\n", interface);
        return 2;	
	}

	while ( (service = IOIteratorNext(iter)) != IO_OBJECT_NULL ) {
		
		kret = IORegistryEntryGetPath(service, kIODeviceTreePlane, pathInPlane);
		if(kret == 0) {
			gotPathInPlane = true;
			IOObjectRelease(service);
			break;		
		}
		
		IOObjectRelease(service);    
	}
	IOObjectRelease(iter);

	if(!gotPathInPlane) {
        contextprintf(context, kBLLogLevelError, "Could not find parent for %s in device tree\n", interface);
		return 3;
	}

	contextprintf(context, kBLLogLevelVerbose, "Got path %s for interface %s\n", pathInPlane, interface);

	if(host && path && strlen(path)) {
		snprintf(ofstring, ofstringSize, "%s:%s,%s", pathInPlane + strlen(kIODeviceTreePlane) + 1, host, path);
	} else {
		snprintf(ofstring, ofstringSize, "%s:bootp", pathInPlane + strlen(kIODeviceTreePlane) + 1);
	}

	return 0;
}
