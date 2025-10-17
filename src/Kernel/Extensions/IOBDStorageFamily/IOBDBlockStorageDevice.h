/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
/*!
 * @header IOBDBlockStorageDevice
 * @abstract
 * This header contains the IOBDBlockStorageDevice class definition.
 */

#ifndef	_IOBDBLOCKSTORAGEDEVICE_H
#define	_IOBDBLOCKSTORAGEDEVICE_H

#include <IOKit/storage/IOBDTypes.h>

/*!
 * @defined kIOBDBlockStorageDeviceClass
 * @abstract
 * kIOBDBlockStorageDeviceClass is the name of the IOBDBlockStorageDevice class.
 * @discussion
 * kIOBDBlockStorageDeviceClass is the name of the IOBDBlockStorageDevice class.
 */

#define kIOBDBlockStorageDeviceClass "IOBDBlockStorageDevice"

#ifdef KERNEL
#ifdef __cplusplus

/*
 * Kernel
 */

#include <IOKit/storage/IODVDBlockStorageDevice.h>

/* Property used for matching, so the generic driver gets the nub it wants. */
#define	kIOBlockStorageDeviceTypeBD	"BD"

/*!
 * @class
 * IOBDBlockStorageDevice
 * @abstract
 * The IOBDBlockStorageDevice class is a generic BD block storage device
 * abstraction.
 * @discussion
 * This class is the protocol for generic BD functionality, independent of
 * the physical connection protocol (e.g. SCSI, ATA, USB).
 *
 * The APIs are the union of CD APIs, DVD APIs, and all
 * necessary new low-level BD APIs.
 *
 * A subclass implements relay methods that translate our requests into
 * calls to a protocol- and device-specific provider.
 */

class __exported IOBDBlockStorageDevice : public IODVDBlockStorageDevice
{
    OSDeclareAbstractStructors(IOBDBlockStorageDevice)

protected:

    struct ExpansionData { /* */ };
    ExpansionData * _expansionData;

public:

    /*!
     * @function init
     * @discussion
     * Initialize this object's minimal state.
     * @param properties
     * Substitute property table for this object (optional).
     * @result
     * Returns true on success, false otherwise.
     */

    virtual bool init(OSDictionary * properties);

    /*!
     * @function readDiscStructure
     * @discussion
     * Issue an MMC READ DISC STRUCTURE command.
     * @param buffer
     * Buffer for the data transfer.  The size of the buffer implies the size of
     * the data transfer.
     * @param format
     * As documented by MMC.
     * @param address
     * As documented by MMC.
     * @param layer
     * As documented by MMC.
     * @param grantID
     * As documented by MMC.
     * @param type
     * As documented by MMC.
     * @result
     * Returns the status of the data transfer.
     */

    virtual IOReturn readDiscStructure( IOMemoryDescriptor * buffer,
                                        UInt8                format,
                                        UInt32               address,
                                        UInt8                layer,
                                        UInt8                grantID,
                                        UInt8                type ) = 0;

    /*!
     * @function splitTrack
     * @discussion
     * Issue an MMC RESERVE TRACK command with the ARSV bit.
     * @param address
     * As documented by MMC.
     * @result
     * Returns the status of the operation.
     */

    virtual IOReturn splitTrack(UInt32 address) = 0;

    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  0);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  1);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  2);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  3);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  4);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  5);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  6);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  7);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  8);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice,  9);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice, 10);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice, 11);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice, 12);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice, 13);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice, 14);
    OSMetaClassDeclareReservedUnused(IOBDBlockStorageDevice, 15);
};

#endif /* __cplusplus */
#endif /* KERNEL */
#endif /* !_IOBDBLOCKSTORAGEDEVICE_H */
