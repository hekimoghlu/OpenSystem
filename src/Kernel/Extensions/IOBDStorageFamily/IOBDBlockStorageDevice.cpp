/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#include <IOKit/storage/IOBDBlockStorageDevice.h>

#define super IODVDBlockStorageDevice
OSDefineMetaClassAndAbstractStructors(IOBDBlockStorageDevice, IODVDBlockStorageDevice)

bool IOBDBlockStorageDevice::init(OSDictionary * properties)
{
    //
    // Initialize this object's minimal state.
    //

    // Ask our superclass' opinion.

    if (super::init(properties) == false)  return false;

    // Create our registry properties.

    setProperty(kIOBlockStorageDeviceTypeKey, kIOBlockStorageDeviceTypeBD);

    return true;
}

OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  0);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  1);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  2);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  3);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  4);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  5);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  6);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  7);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  8);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice,  9);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice, 10);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice, 11);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice, 12);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice, 13);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice, 14);
OSMetaClassDefineReservedUnused(IOBDBlockStorageDevice, 15);
