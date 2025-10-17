/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#if defined(__i386__) || defined(__x86_64__)

#include <IOKit/system.h>

#include <IOKit/pci/IOPCIBridge.h>
#include <IOKit/pci/IOPCIDevice.h>
#include <IOKit/IOLib.h>
#include <IOKit/assert.h>
#include <libkern/c++/OSContainers.h>

#include <architecture/i386/pio.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

UInt32 IOPCIDevice::ioRead32( UInt16 offset, IOMemoryMap * map )
{
    UInt32      value;

    if (0 == map)
        map = ioMap;

    /*
     * getPhysicalAddress() can block on a mutex. Since I/O memory
     * ranges behaves identity mapped, switch to getVirtualAddress().
     */
    value = inl( map->getVirtualAddress() + offset );

    return (value);
}

UInt16 IOPCIDevice::ioRead16( UInt16 offset, IOMemoryMap * map )
{
    UInt16      value;

    if (0 == map)
        map = ioMap;

    value = inw( map->getVirtualAddress() + offset );

    return (value);
}

UInt8 IOPCIDevice::ioRead8( UInt16 offset, IOMemoryMap * map )
{
    UInt32      value;

    if (0 == map)
        map = ioMap;

    value = inb( map->getVirtualAddress() + offset );

    return (value);
}

void IOPCIDevice::ioWrite32( UInt16 offset, UInt32 value,
                             IOMemoryMap * map )
{
    if (0 == map)
        map = ioMap;

    outl( map->getVirtualAddress() + offset, value );
}

void IOPCIDevice::ioWrite16( UInt16 offset, UInt16 value,
                             IOMemoryMap * map )
{
    if (0 == map)
        map = ioMap;

    outw( map->getVirtualAddress() + offset, value );
}

void IOPCIDevice::ioWrite8( UInt16 offset, UInt8 value,
                            IOMemoryMap * map )
{
    if (0 == map)
        map = ioMap;

    outb( map->getVirtualAddress() + offset, value );
}


#endif // __i386__
