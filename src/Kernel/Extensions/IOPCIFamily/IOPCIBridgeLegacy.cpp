/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#include <IOKit/pci/IOPCIPrivate.h>

#if ACPI_SUPPORT

#include <IOKit/acpi/IOACPIPlatformDevice.h>
#include <i386/cpuid.h>
#include <i386/cpu_number.h>
#include "AppleVTD.h"

const IORegistryPlane*                   gIOPCIACPIPlane;
const OSSymbol*                          gIOPCIPSMethods[kIOPCIDevicePowerStateCount];
extern IOPCIMessagedInterruptController* gIOPCIMessagedInterruptController;

#define kACPITablesKey                      "ACPI Tables"
enum
{
    // LAPIC_DEFAULT_INTERRUPT_BASE (mp.h)
    kBaseMessagedInterruptVectors = 0x70,
    kNumMessagedInterruptVectors  = 0x200 - kBaseMessagedInterruptVectors
};

// Shared data for all host bridges
IOPCIHostBridgeData *gBridgeData = NULL;

void InitSharedBridgeData(void)
{
	if (gBridgeData == NULL)
	{
		gBridgeData = OSTypeAlloc(IOPCIHostBridgeData);
		if (gBridgeData == NULL || gBridgeData->init() == false)
		{
			panic("Failed to initialize global host bridge data structure");
		}
	}
}

IOReturn IOPCIPlatformInitialize(void)
{
    IOPCIMessagedInterruptController* ic;
    OSDictionary*                     dict;
    OSObject*                         obj;
    OSData*                           data;
    IOService*                        provider;
    bool                              ok;
    int                               lapic_max_interrupt_cpunum;
    int                               forceintcpu;

    data     = 0;
    ok       = true;
    provider = IOService::getPlatform();
    obj      = provider->copyProperty(kACPITablesKey);
    if((dict = OSDynamicCast(OSDictionary, obj)))
    {
        data = OSDynamicCast(OSData, dict->getObject("DMAR"));
    }

    if(!PE_parse_boot_argn("intcpumax", &lapic_max_interrupt_cpunum, sizeof(lapic_max_interrupt_cpunum)))
    {
        lapic_max_interrupt_cpunum = ((cpuid_features() & CPUID_FEATURE_HTT) ? 1 : 0);
    }
    if(!PE_parse_boot_argn("forceintcpu", &forceintcpu, sizeof(forceintcpu)))
    {
        forceintcpu = 0;
    }

    ic = new IOPCIMessagedInterruptController;
    if(ic && !ic->init(kNumMessagedInterruptVectors, kBaseMessagedInterruptVectors))
    {
        OSSafeReleaseNULL(ic);
    }
    if(ic)
    {
        if(forceintcpu && lapic_max_interrupt_cpunum)
        {
            // no vectors below 0x100
            ok = ic->reserveVectors(0, 0x100 - kBaseMessagedInterruptVectors);
        }
        else
        {
            // used by osfmk/cpu
            ok  = ic->reserveVectors(0x7F - kBaseMessagedInterruptVectors, 4);
            ok &= ic->reserveVectors(0xD0 - kBaseMessagedInterruptVectors, 16);
            ok &= ic->reserveVectors(0xFF - kBaseMessagedInterruptVectors, 1);
        }
        if(lapic_max_interrupt_cpunum)
        {
            // used by osfmk/cpu
            ok &= ic->reserveVectors(0x100 - kBaseMessagedInterruptVectors, 32);
            ok &= ic->reserveVectors(0x17F - kBaseMessagedInterruptVectors, 4);
            ok &= ic->reserveVectors(0x1D0 - kBaseMessagedInterruptVectors, 16);
            ok &= ic->reserveVectors(0x1FF - kBaseMessagedInterruptVectors, 1);
        }
        else
        {
            // no vectors above 0xFF
            ok = ic->reserveVectors(0x100 - kBaseMessagedInterruptVectors, 0x100);
        }
    }
    if(!ic || !ok) panic("IOPCIMessagedInterruptController");

    InitSharedBridgeData();
	AppleVTD::install(gBridgeData->_configWorkLoop, gIOPCIFlags, provider, data, ic);

    if(obj) obj->release();

    return kIOReturnSuccess;
}

IOReturn IOPCIBridge::callPlatformFunction(const OSSymbol* functionName,
                                           bool waitForFunction,
                                           void* p1, void* p2,
                                           void* p3, void* p4)
{
    IOReturn result;

    result = IOService::callPlatformFunction(functionName, waitForFunction,
                                             p1, p2, p3, p4);

    if((kIOReturnUnsupported == result)
       && (gIOPlatformGetMessagedInterruptControllerKey == functionName))
    {
        *(IOPCIMessagedInterruptController**)p2 = gIOPCIMessagedInterruptController;
    }

    return result;
}

#endif /* ACPI_SUPPORT */
