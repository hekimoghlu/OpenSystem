/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
 * @header IOBDMedia
 * @abstract
 * This header contains the IOBDMedia class definition.
 */

#ifndef _IOBDMEDIA_H
#define _IOBDMEDIA_H

/*!
 * @defined kIOBDMediaClass
 * @abstract
 * kIOBDMediaClass is the name of the IOBDMedia class.
 * @discussion
 * kIOBDMediaClass is the name of the IOBDMedia class.
 */

#define kIOBDMediaClass "IOBDMedia"

/*!
 * @defined kIOBDMediaTypeKey
 * @abstract
 * kIOBDMediaTypeKey is a property of IOBDMedia objects.  It has an OSString
 * value.
 * @discussion
 * The kIOBDMediaTypeKey property identifies the BD media type (BD-ROM, BD-R,
 * BD-RE, etc).  See the kIOBDMediaType contants for possible values.
 */

#define kIOBDMediaTypeKey "Type"

/*!
 * @defined kIOBDMediaTypeROM
 * The kIOBDMediaTypeKey constant for BD-ROM media.
 */

#define kIOBDMediaTypeROM "BD-ROM"

/*!
 * @defined kIOBDMediaTypeR
 * The kIOBDMediaTypeKey constant for BD-R media.
 */

#define kIOBDMediaTypeR "BD-R"

/*!
 * @defined kIOBDMediaTypeRE
 * The kIOBDMediaTypeKey constant for BD-RE media.
 */

#define kIOBDMediaTypeRE "BD-RE"

#ifdef KERNEL
#ifdef __cplusplus

/*
 * Kernel
 */

#include <IOKit/storage/IOBDBlockStorageDriver.h>
#include <IOKit/storage/IOMedia.h>

/*!
 * @class IOBDMedia
 * @abstract
 * The IOBDMedia class is a random-access disk device abstraction for BDs.
 * @discussion
 * The IOBDMedia class is a random-access disk device abstraction for BDs.
 */

class __exported IOBDMedia : public IOMedia
{
    OSDeclareDefaultStructors(IOBDMedia)

protected:

    struct ExpansionData { /* */ };
    ExpansionData * _expansionData;

public:

    /*
     * Obtain this object's provider.   We override the superclass's method to
     * return a more specific subclass of IOService -- IOBDBlockStorageDriver.
     * This method serves simply as a convenience to subclass developers.
     */

    virtual IOBDBlockStorageDriver * getProvider() const;

    /*
     * Compare the properties in the supplied table to this object's properties.
     */

    virtual bool matchPropertyTable(OSDictionary * table, SInt32 * score);

    /*
     * Issue an MMC REPORT KEY command.
     * Obsoleted, replaced by this interface.
     *     virtual IOReturn reportKey( IOMemoryDescriptor * buffer,
     *                                 UInt8                keyClass,
     *                                 UInt32               address,
     *                                 UInt8                blockCount,
     *                                 UInt8                grantID,
     *                                 UInt8                format );
     */

    virtual IOReturn reportKey( IOMemoryDescriptor * buffer,
                                UInt8                keyClass,
                                UInt32               address,
                                UInt8                grantID,
                                UInt8                format ) __attribute__ ((deprecated));

    /*!
     * @function sendKey
     * @discussion
     * Issue an MMC SEND KEY command.
     * @param buffer
     * Buffer for the data transfer.  The size of the buffer implies the size of
     * the data transfer.
     * @param keyClass
     * As documented by MMC.
     * @param grantID
     * As documented by MMC.
     * @param format
     * As documented by MMC.
     * @result
     * Returns the status of the data transfer.
     */

    virtual IOReturn sendKey( IOMemoryDescriptor * buffer,
                              UInt8                keyClass,
                              UInt8                grantID,
                              UInt8                format );

    /*!
     * @function readStructure
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
     * @result
     * Returns the status of the data transfer.
     */

    virtual IOReturn readStructure( IOMemoryDescriptor *     buffer,
                                    UInt8                    format,
                                    UInt32                   address,
                                    UInt8                    layer,
                                    UInt8                    grantID );

    /*!
     * @function getSpeed
     * @discussion
     * Get the current speed used for data transfers.
     * @param kilobytesPerSecond
     * Returns the current speed used for data transfers, in kB/s.
     *
     * kBDSpeedMin specifies the minimum speed for all BD media (1X).
     * kBDSpeedMax specifies the maximum speed supported in hardware.
     * @result
     * Returns the status of the operation.
     */

    virtual IOReturn getSpeed(UInt16 * kilobytesPerSecond);

    /*!
     * @function setSpeed
     * @discussion
     * Set the speed to be used for data transfers.
     * @param kilobytesPerSecond
     * Speed to be used for data transfers, in kB/s.
     *
     * kBDSpeedMin specifies the minimum speed for all BD media (1X).
     * kBDSpeedMax specifies the maximum speed supported in hardware.
     * @result
     * Returns the status of the operation.
     */

    virtual IOReturn setSpeed(UInt16 kilobytesPerSecond);

    /*!
     * @function readDiscInfo
     * @discussion
     * Issue an MMC READ DISC INFORMATION command.
     * @param buffer
     * Buffer for the data transfer.  The size of the buffer implies the size of
     * the data transfer.
     * @param type
     * Reserved for future use.  Set to zero.
     * @param actualByteCount
     * Returns the actual number of bytes transferred in the data transfer.
     * @result
     * Returns the status of the data transfer.
     */

    virtual IOReturn readDiscInfo( IOMemoryDescriptor * buffer,
                                   UInt8                type,
                                   UInt16 *             actualByteCount );

    /*!
     * @function readTrackInfo
     * @discussion
     * Issue an MMC READ TRACK INFORMATION command.
     * @param buffer
     * Buffer for the data transfer.  The size of the buffer implies the size of
     * the data transfer.
     * @param address
     * As documented by MMC.
     * @param addressType
     * As documented by MMC.
     * @param open
     * Reserved for future use.  Set to zero.
     * @param actualByteCount
     * Returns the actual number of bytes transferred in the data transfer.
     * @result
     * Returns the status of the data transfer.
     */

    virtual IOReturn readTrackInfo( IOMemoryDescriptor * buffer,
                                    UInt32               address,
                                    UInt8                addressType,
                                    UInt8                open,
                                    UInt16 *             actualByteCount );

    /*!
     * @function splitTrack
     * @discussion
     * Issue an MMC RESERVE TRACK command with the ARSV bit.
     * @param address
     * As documented by MMC.
     * @result
     * Returns the status of the operation.
     */

    virtual IOReturn splitTrack(UInt32 address);

    /*!
     * @function reportKey
     * @discussion
     * Issue an MMC REPORT KEY command.
     * @param buffer
     * Buffer for the data transfer.  The size of the buffer implies the size of
     * the data transfer.
     * @param keyClass
     * As documented by MMC.
     * @param address
     * As documented by MMC.
     * @param blockCount
     * As documented by MMC.
     * @param grantID
     * As documented by MMC.
     * @param format
     * As documented by MMC.
     * @result
     * Returns the status of the data transfer.
     */

    virtual IOReturn reportKey( IOMemoryDescriptor * buffer,
                                UInt8                keyClass,
                                UInt32               address,
                                UInt8                blockCount,
                                UInt8                grantID,
                                UInt8                format );

    OSMetaClassDeclareReservedUsed(IOBDMedia,  0);		/* reportKey */
    OSMetaClassDeclareReservedUnused(IOBDMedia,  1);
    OSMetaClassDeclareReservedUnused(IOBDMedia,  2);
    OSMetaClassDeclareReservedUnused(IOBDMedia,  3);
    OSMetaClassDeclareReservedUnused(IOBDMedia,  4);
    OSMetaClassDeclareReservedUnused(IOBDMedia,  5);
    OSMetaClassDeclareReservedUnused(IOBDMedia,  6);
    OSMetaClassDeclareReservedUnused(IOBDMedia,  7);
    OSMetaClassDeclareReservedUnused(IOBDMedia,  8);
    OSMetaClassDeclareReservedUnused(IOBDMedia,  9);
    OSMetaClassDeclareReservedUnused(IOBDMedia, 10);
    OSMetaClassDeclareReservedUnused(IOBDMedia, 11);
    OSMetaClassDeclareReservedUnused(IOBDMedia, 12);
    OSMetaClassDeclareReservedUnused(IOBDMedia, 13);
    OSMetaClassDeclareReservedUnused(IOBDMedia, 14);
    OSMetaClassDeclareReservedUnused(IOBDMedia, 15);
};

#endif /* __cplusplus */
#endif /* KERNEL */
#endif /* !_IOBDMEDIA_H */
