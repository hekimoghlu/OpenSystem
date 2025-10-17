/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#include <CoreFoundation/CoreFoundation.h>

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/mach_error.h>

#include <libc.h>
#include <servers/bootstrap.h>
#include <sysexits.h>

#include <IOKit/IOKitLib.h>
#include <IOKit/IOKitServer.h>
#include <IOKit/IOCFURLAccess.h>
#include <IOKit/IOCFSerialize.h>
#include <IOKit/IOCFUnserialize.h>
#include <IOKit/IOMessage.h>
#include <IOKit/ps/IOPSKeys.h>

#include "IOUPSPrivate.h"
#include <IOKit/ps/IOUPSPlugIn.h>

// mig generated header
#include "ioupspluginmig.h"

Boolean IOUPSMIGServerIsRunning(mach_port_t * bootstrap_port_ref, mach_port_t * upsd_port_ref)
{
    mach_port_t     active = MACH_PORT_NULL;
    kern_return_t   kern_result = KERN_SUCCESS;
    mach_port_t     bootstrap_port;

    if (bootstrap_port_ref && (*bootstrap_port_ref != MACH_PORT_NULL)) {
        bootstrap_port = *bootstrap_port_ref;
    } else {
        /* Get the bootstrap server port */
        kern_result = task_get_bootstrap_port(mach_task_self(), &bootstrap_port);
        if (kern_result != KERN_SUCCESS) {
            return false;
        }
        if (bootstrap_port_ref) {
            *bootstrap_port_ref = bootstrap_port;
        }
    }

    /* Check "upsd" server status */
    kern_result = bootstrap_look_up(
                        bootstrap_port, 
                        kIOUPSPlugInServerName, 
                        &active);

    if (BOOTSTRAP_SUCCESS == kern_result) {
        return true;
    } else {
        // For any result other than SUCCESS, we presume the server is
        // not running. We expect the most common failure result to be:
        // kern_result == BOOTSTRAP_UNKNOWN_SERVICE
        return false;
    }
}

IOReturn IOUPSSendCommand(mach_port_t connect, int upsID, CFDictionaryRef command)
{
    IOReturn 		ret;
    CFDataRef		serializedData;

    if (!connect || !command)
        return kIOReturnBadArgument;

    serializedData = (CFDataRef)IOCFSerialize( command, kNilOptions );

    if (!serializedData)
        return kIOReturnError;
        
    ret = io_ups_send_command(connect, upsID, 
                (vm_offset_t)CFDataGetBytePtr(serializedData), 
                (mach_msg_type_number_t) CFDataGetLength(serializedData));
        
    CFRelease( serializedData );

    return ret;
}

IOReturn IOUPSGetEvent(mach_port_t connect, int upsID, CFDictionaryRef *event)
{
    IOReturn        ret;
    void *          buffer = NULL;
    IOByteCount     bufferSize;

    if (!connect || !event)
        return kIOReturnBadArgument;

    ret = io_ups_get_event(connect, upsID, 
                (vm_offset_t *)&buffer, 
                (mach_msg_type_number_t *)&bufferSize);
    
    if ( ret != kIOReturnSuccess )
        return ret;

    *event = IOCFUnserialize(buffer, kCFAllocatorDefault, kNilOptions, NULL);

    vm_deallocate(mach_task_self(), (vm_address_t)buffer, bufferSize);
    
    return ret;
}

IOReturn IOUPSGetCapabilities(mach_port_t connect, int upsID, CFSetRef *capabilities)
{
    IOReturn 		ret;
    void *		buffer = NULL;
    IOByteCount		bufferSize;

    if (!connect || !capabilities)
        return kIOReturnBadArgument;

    ret = io_ups_get_capabilities(connect, upsID, 
                (vm_offset_t *)&buffer, 
                (mach_msg_type_number_t *)&bufferSize);
    
    if ( ret != kIOReturnSuccess )
        return ret;

    *capabilities = IOCFUnserialize(buffer, kCFAllocatorDefault, kNilOptions, NULL);

    vm_deallocate(mach_task_self(), (vm_address_t)buffer, bufferSize);

    return ret;
}

