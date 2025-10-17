/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include <IOKit/storage/IODVDBlockStorageDevice.h>

#define	super	IOCDBlockStorageDevice
OSDefineMetaClassAndAbstractStructors(IODVDBlockStorageDevice,IOCDBlockStorageDevice)

extern "C" IOReturn _ZN23IODVDBlockStorageDevice9reportKeyEP18IOMemoryDescriptorhjhhh (
                                        IOMemoryDescriptor *buffer,const DVDKeyClass keyClass,
                                        const UInt32 lba,const UInt8 blockCount,
                                        const UInt8 agid,const DVDKeyFormat keyFormat);

#define dvdStorageReportKeyWithBlockCount( dvdStorage )                                                   \
( OSMemberFunctionCast( void *,                                                                           \
                        dvdStorage,                                                                       \
                        ( IOReturn ( IODVDBlockStorageDevice::* )( IOMemoryDescriptor *,                  \
                                                                   const DVDKeyClass,                     \
                                                                   const UInt32, const UInt8,             \
                                                                   const UInt8 agid, const DVDKeyFormat ) \
                        ) &IODVDBlockStorageDevice::reportKey ) !=                                        \
  _ZN23IODVDBlockStorageDevice9reportKeyEP18IOMemoryDescriptorhjhhh )

bool
IODVDBlockStorageDevice::init(OSDictionary * properties)
{
    bool result;

    result = super::init(properties);
    if (result) {
        setProperty(kIOBlockStorageDeviceTypeKey,kIOBlockStorageDeviceTypeDVD);
    }

    return(result);
}

IOReturn
IODVDBlockStorageDevice::reportKey(IOMemoryDescriptor *buffer,const DVDKeyClass keyClass,
                                        const UInt32 lba,const UInt8 agid,const DVDKeyFormat keyFormat)
{
    if ( dvdStorageReportKeyWithBlockCount ( this ) )
    {
        return(reportKey(buffer,keyClass,lba,0,agid,keyFormat));
    }
    return kIOReturnUnsupported;
}

IOReturn
IODVDBlockStorageDevice::reportKey(IOMemoryDescriptor *buffer,const DVDKeyClass keyClass,
                                        const UInt32 lba,const UInt8 blockCount,
                                        const UInt8 agid,const DVDKeyFormat keyFormat)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

    if ( blockCount == 0 )
    {
        return ( reportKey( buffer,keyClass,lba,agid,keyFormat ) );
    }
    return kIOReturnUnsupported;

#pragma clang diagnostic pop
}

OSMetaClassDefineReservedUsed(IODVDBlockStorageDevice,  0);     /* reportKey */
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  1);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  2);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  3);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  4);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  5);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  6);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  7);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  8);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice,  9);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 10);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 11);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 12);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 13);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 14);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 15);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 16);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 17);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 18);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 19);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 20);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 21);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 22);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 23);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 24);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 25);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 26);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 27);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 28);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 29);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 30);
OSMetaClassDefineReservedUnused(IODVDBlockStorageDevice, 31);
