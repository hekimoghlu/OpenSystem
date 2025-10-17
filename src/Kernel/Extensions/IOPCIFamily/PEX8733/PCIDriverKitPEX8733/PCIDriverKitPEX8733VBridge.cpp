/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 31, 2025.
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

//
//  PCIDriverKitPEX8733VBridge.cpp
//  PCIDriverKitPEX8733VBridge
//
//  Created by Kevin Strasberg on 10/29/19.
//
#include <stdio.h>
#include <os/log.h>
#include <DriverKit/DriverKit.h>
#include <PCIDriverKit/PCIDriverKit.h>

#include "PCIDriverKitPEX8733VBridge.h"
#include "PEX8733Definitions.h"
#define debugLog(fmt, args...)  os_log(OS_LOG_DEFAULT, "PCIDriverKitPEX8733VBridge::%s:  " fmt,  __FUNCTION__,##args)

struct PCIDriverKitPEX8733VBridge_IVars
{
    IOPCIDevice* pciDevice;
    uint64_t     vendorCapabilityID;
};

bool
PCIDriverKitPEX8733VBridge::init()
{
    if(super::init() != true)
    {
        return false;
    }

    ivars = IONewZero(PCIDriverKitPEX8733VBridge_IVars, 1);

    if(ivars == NULL)
    {
        return false;
    }

    return true;
}

void
PCIDriverKitPEX8733VBridge::free()
{
    IOSafeDeleteNULL(ivars, PCIDriverKitPEX8733VBridge_IVars, 1);
    super::free();
}

kern_return_t
IMPL(PCIDriverKitPEX8733VBridge, Start)
{
    kern_return_t result = Start(provider, SUPERDISPATCH);

    if(result != kIOReturnSuccess)
    {
        return result;
    }

    ivars->pciDevice = OSDynamicCast(IOPCIDevice, provider);
    if(ivars->pciDevice == NULL)
    {
        Stop(provider);
        return kIOReturnNoDevice;
    }
    ivars->pciDevice->retain();

    if(ivars->pciDevice->Open(this, 0) != kIOReturnSuccess)
    {
        Stop(provider);
        return kIOReturnNoDevice;
    }

    debugLog("enabling bus lead and memory space");
    uint16_t command;
    ivars->pciDevice->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &command);
    ivars->pciDevice->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand,
                                           command | kIOPCICommandBusLead | kIOPCICommandMemorySpace);

    return result;
}

kern_return_t
IMPL(PCIDriverKitPEX8733VBridge, Stop)
{
    debugLog("disabling bus lead and memory space");
    if(ivars->pciDevice != NULL)
    {
        uint16_t command;
        ivars->pciDevice->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &command);
        ivars->pciDevice->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand,
                                               command & ~(kIOPCICommandBusLead | kIOPCICommandMemorySpace));

    }
    OSSafeReleaseNULL(ivars->pciDevice);
    return Stop(provider, SUPERDISPATCH);
}
