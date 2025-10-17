/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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
#ifndef _IOKIT_PM_IOUPSPRIVATE_H
#define _IOKIT_PM_IOUPSPRIVATE_H

#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <IOKit/IOReturn.h>

/*!
    @defined kIOUPSDeviceKey
    @abstract Key for IOService object that denotes a UPS device.
    @discussion It is expected that every IOService module that contains
    a IOUPSCFPlugIn will at least define this key in it property table.
*/
#define kIOUPSDeviceKey             "UPSDevice"
#define kIOPowerDeviceUsageKey      0x84
#define kIOBatterySystemUsageKey    0x85

#define kIOPSPrivateBatteryCaseType "Battery Case"

/*!
    @defined kIOUPSPlugInServerName
    @abstract Key for UPS Mig server.
    @discussion Used for identifying UPS mig server.
*/
#define kIOUPSPlugInServerName		"com.apple.IOUPSPlugInServer" 

#define MAKE_UNIQ_SOURCE_ID(pid, psID)      (((pid & 0xffff) << 16) | (psID & 0xffff))
#define GET_UPSID(srcId)                    (srcId & 0xffff)

Boolean IOUPSMIGServerIsRunning(mach_port_t * bootstrap_port_ref, mach_port_t * upsd_port_ref);

IOReturn IOUPSSendCommand(mach_port_t connect, int upsID, CFDictionaryRef command);

IOReturn IOUPSGetEvent(mach_port_t connect, int upsID, CFDictionaryRef *event);

IOReturn IOUPSGetCapabilities(mach_port_t connect, int upsID, CFSetRef *capabilities);

#endif /* !_IOKIT_PM_IOUPSPRIVATE_H */
