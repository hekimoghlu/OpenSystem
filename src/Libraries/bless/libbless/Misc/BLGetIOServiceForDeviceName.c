/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
 *  BLGetIOServiceForDeviceName.c
 *  bless
 *
 *  Created by Shantonu Sen on 2/7/05.
 *  Copyright 2005-2007 Apple Inc. All Rights Reserved.
 *
 *  $Id: BLGetIOServiceForDeviceName.c,v 1.4 2006/02/20 22:49:56 ssen Exp $
 *
 */

#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/param.h>
#include <sys/mount.h>

#import <mach/mach_error.h>

#import <IOKit/IOKitLib.h>
#import <IOKit/IOBSD.h>

#include <CoreFoundation/CoreFoundation.h>

#include "bless.h"
#include "bless_private.h"

int BLGetIOServiceForDeviceName(BLContextPtr context, const char * devName,
								io_service_t *service)
{
    io_service_t			myservice;
    io_iterator_t			services;
    kern_return_t			kret;
	mach_port_t				ourIOKitPort;

	*service = 0;
    
	// Obtain the I/O Kit communication handle.
    if((kret = IOMasterPort(bootstrap_port, &ourIOKitPort)) != KERN_SUCCESS) {
		return 1;
    }
	
	
    kret = IOServiceGetMatchingServices(ourIOKitPort,
										IOBSDNameMatching(ourIOKitPort,
														  0,
														  devName),
										&services);
    if (kret != KERN_SUCCESS) {
        return 2;
    }
    
    // Should only be one IOKit object for this volume. (And we only want one.)
    myservice = IOIteratorNext(services);
    if (!myservice) {
        IOObjectRelease(services);
        return 4;
    }
    
    IOObjectRelease(services);
    
	*service = myservice;
	
    return 0;
}
