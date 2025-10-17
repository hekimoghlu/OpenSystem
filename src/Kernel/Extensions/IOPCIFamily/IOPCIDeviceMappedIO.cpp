/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#if !defined(__i386__) && !defined(__x86_64__)

#include <IOKit/system.h>

#include <IOKit/pci/IOPCIBridge.h>
#include <IOKit/pci/IOPCIDevice.h>

#include <IOKit/IOLib.h>
#include <IOKit/assert.h>

#include <libkern/OSByteOrder.h>
#include <libkern/c++/OSContainers.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

UInt32 IOPCIDevice::ioRead32( UInt16 offset, IOMemoryMap * map )
{
    UInt32      value;

    if (0 == map)
    {
        map = ioMap;
        if (0 == map)
            return (0);
    }

    value = OSReadLittleInt32( (volatile void *)map->getVirtualAddress(), offset);
    OSSynchronizeIO();

    return (value);
}

UInt16 IOPCIDevice::ioRead16( UInt16 offset, IOMemoryMap * map )
{
    UInt16      value;

    if (0 == map)
    {
        map = ioMap;
        if (0 == map)
            return (0);
    }

    value = OSReadLittleInt16( (volatile void *)map->getVirtualAddress(), offset);
    OSSynchronizeIO();

    return (value);
}

UInt8 IOPCIDevice::ioRead8( UInt16 offset, IOMemoryMap * map )
{
    UInt32      value;

    if (0 == map)
    {
        map = ioMap;
        if (0 == map)
            return (0);
    }

    value = ((volatile UInt8 *) map->getVirtualAddress())[ offset ];
    OSSynchronizeIO();

    return (value);
}

void IOPCIDevice::ioWrite32( UInt16 offset, UInt32 value,
                             IOMemoryMap * map )
{
    if (0 == map)
    {
        map = ioMap;
        if (0 == map)
            return ;
    }

    OSWriteLittleInt32( (volatile void *)map->getVirtualAddress(), offset, value);
    OSSynchronizeIO();
}

void IOPCIDevice::ioWrite16( UInt16 offset, UInt16 value,
                             IOMemoryMap * map )
{
    if (0 == map)
    {
        map = ioMap;
        if (0 == map)
            return ;
    }

    OSWriteLittleInt16( (volatile void *)map->getVirtualAddress(), offset, value);
    OSSynchronizeIO();
}

void IOPCIDevice::ioWrite8( UInt16 offset, UInt8 value,
                            IOMemoryMap * map )
{
    if (0 == map)
    {
        map = ioMap;
        if (0 == map)
            return ;
    }

    ((volatile UInt8 *) map->getVirtualAddress())[ offset ] = value;
    OSSynchronizeIO();
}

#endif //  !defined(__i386__) && !defined(__x86_64__)


