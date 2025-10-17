/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 4, 2025.
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
#include <IOKit/storage/IOCDBlockStorageDevice.h>

#define	super	IOBlockStorageDevice
OSDefineMetaClassAndAbstractStructors(IOCDBlockStorageDevice,IOBlockStorageDevice)

bool
IOCDBlockStorageDevice::init(OSDictionary * properties)
{
    bool result;

    result = super::init(properties);
    if (result) {
        setProperty(kIOBlockStorageDeviceTypeKey,
                    kIOBlockStorageDeviceTypeCDROM);
    }

    return(result);
}

OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  0);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  1);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  2);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  3);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  4);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  5);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  6);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  7);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  8);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice,  9);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice, 10);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice, 11);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice, 12);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice, 13);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice, 14);
OSMetaClassDefineReservedUnused(IOCDBlockStorageDevice, 15);
