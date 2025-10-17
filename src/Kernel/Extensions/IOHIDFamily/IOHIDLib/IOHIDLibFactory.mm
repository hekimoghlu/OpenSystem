/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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
#include <TargetConditionals.h>

#include <IOKit/hid/IOHIDDevicePlugIn.h>
#include <IOKit/hid/IOHIDServicePlugIn.h>
#include "IOHIDIUnknown.h"
#import "IOHIDDeviceClass.h"
#import "IOHIDObsoleteDeviceClass.h"
#import "IOHIDUPSClass.h"
#include "IOHIDEventServiceFastPathClass.h"

extern "C" void *IOHIDLibFactory(CFAllocatorRef allocator __unused, CFUUIDRef typeID);

void *IOHIDLibFactory(CFAllocatorRef allocator __unused, CFUUIDRef typeID)
{
    IOCFPlugInInterface **interface = NULL;
    
    if (CFEqual(typeID, kIOHIDDeviceUserClientTypeID)) {
        IOHIDObsoleteDeviceClass *device = [[IOHIDObsoleteDeviceClass alloc] init];
        [device queryInterface:CFUUIDGetUUIDBytes(kIOHIDDeviceInterfaceID)
               outInterface:(LPVOID *)&interface];
    } else if (CFEqual(typeID, kIOHIDDeviceTypeID)) {
        IOHIDDeviceClass *device = [[IOHIDDeviceClass alloc] init];
        [device queryInterface:CFUUIDGetUUIDBytes(kIOHIDDeviceDeviceInterfaceID)
                  outInterface:(LPVOID *)&interface];
    } else if (CFEqual(typeID, kIOUPSPlugInTypeID)) {
        IOHIDUPSClass *ups = [[IOHIDUPSClass alloc] init];
        [ups queryInterface:CFUUIDGetUUIDBytes(kIOCFPlugInInterfaceID)
               outInterface:(LPVOID *)&interface];
    } else if (CFEqual(typeID, kIOHIDServiceFastPathPlugInTypeID)) {
        interface = IOHIDEventServiceFastPathClass::alloc();
    }
    
    return (void *)interface;
}
