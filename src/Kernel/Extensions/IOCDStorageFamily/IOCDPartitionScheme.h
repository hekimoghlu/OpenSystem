/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
/*
 * @header IOCDPartitionScheme
 * @abstract
 * This header contains the IOCDPartitionScheme class definition.
 */

#ifndef _IOCDPARTITIONSCHEME_H
#define _IOCDPARTITIONSCHEME_H

#include <IOKit/storage/IOCDTypes.h>

/*
 * @defined kIOCDPartitionSchemeClass
 * @abstract
 * kIOCDPartitionSchemeClass is the name of the IOCDPartitionScheme class.
 * @discussion
 * kIOCDPartitionSchemeClass is the name of the IOCDPartitionScheme class.
 */

#define kIOCDPartitionSchemeClass "IOCDPartitionScheme"

/*
 * @defined kIOMediaSessionIDKey
 * @abstract
 * kIOMediaSessionIDKey is property of IOMedia objects.  It has an OSNumber
 * value.
 * @discussion
 * The kIOMediaSessionIDKey property is placed into each IOMedia instance
 * created by the CD partition scheme.  It identifies the session number
 * the track was recorded on.
 */

#define kIOMediaSessionIDKey "Session ID"

#ifdef KERNEL
#ifdef __cplusplus

/*
 * Kernel
 */

#include <IOKit/storage/IOCDMedia.h>
#include <IOKit/storage/IOPartitionScheme.h>

/*
 * Class
 */

class IOCDPartitionScheme : public IOPartitionScheme
{
    OSDeclareDefaultStructors(IOCDPartitionScheme);

protected:

    struct ExpansionData { /* */ };
    ExpansionData * _expansionData;

    OSSet * _partitions;    /* (set of media objects representing partitions) */

    /*
     * Free all of this object's outstanding resources.
     */

    virtual void free(void);

    /*
     * Scan the provider media for CD partitions (in TOC).  Returns the set
     * of media objects representing each of the partitions (the retain for
     * the set is passed to the caller), or null should no CD partitions be
     * found.  The default probe score can be adjusted up or down, based on
     * the confidence of the scan.
     */

    virtual OSSet * scan(SInt32 * score);

    /*
     * Ask whether the given partition appears to be corrupt.  A partition that
     * is corrupt will cause the failure of the CD partition scheme altogether.
     */

    virtual bool isPartitionCorrupt( CDTOCDescriptor * partition,
                                     UInt64            partitionSize,
                                     UInt32            partitionBlockSize,
                                     CDSectorType      partitionBlockType,
                                     CDTOC *           toc );

    /*
     * Ask whether the given partition appears to be invalid.  A partition that
     * is invalid will cause it to be skipped in the scan, but will not cause a
     * failure of the CD partition scheme.
     */

    virtual bool isPartitionInvalid( CDTOCDescriptor * partition,
                                     UInt64            partitionSize,
                                     UInt32            partitionBlockSize,
                                     CDSectorType      partitionBlockType,
                                     CDTOC *           toc );

    /*
     * Instantiate a new media object to represent the given partition.
     */

    virtual IOMedia * instantiateMediaObject(
                                           CDTOCDescriptor * partition,
                                           UInt64            partitionSize,
                                           UInt32            partitionBlockSize,
                                           CDSectorType      partitionBlockType,
                                           CDTOC *           toc );

    /*
     * Allocate a new media object (called from instantiateMediaObject).
     */

    virtual IOMedia * instantiateDesiredMediaObject(
                                           CDTOCDescriptor * partition,
                                           UInt64            partitionSize,
                                           UInt32            partitionBlockSize,
                                           CDSectorType      partitionBlockType,
                                           CDTOC *           toc );

public:

    /*
     * Initialize this object's minimal state.
     */

    virtual bool init(OSDictionary * properties = 0);

    /*
     * Scan the provider media for CD partitions.
     */

    virtual IOService * probe(IOService * provider, SInt32 * score);

    /*
     * Determine whether the provider media contains CD partitions.
     */

    virtual bool start(IOService * provider);

    /*
     * Read data from the storage object at the specified byte offset into the
     * specified buffer, asynchronously.   When the read completes, the caller
     * will be notified via the specified completion action.
     *
     * The buffer will be retained for the duration of the read.
     *
     * For the CD partition scheme, we convert the read from a partition
     * object into the appropriate readCD command to our provider media.
     */

    virtual void read(IOService *           client,
                      UInt64                byteStart,
                      IOMemoryDescriptor *  buffer,
                      IOStorageAttributes * attributes,
                      IOStorageCompletion * completion);

    /*
     * Write data into the storage object at the specified byte offset from the
     * specified buffer, asynchronously.   When the write completes, the caller
     * will be notified via the specified completion action.
     *
     * The buffer will be retained for the duration of the write.
     *
     * For the CD partition scheme, we convert the write from a partition
     * object into the appropriate writeCD command to our provider media.
     */

    virtual void write(IOService *           client,
                       UInt64                byteStart,
                       IOMemoryDescriptor *  buffer,
                       IOStorageAttributes * attributes,
                       IOStorageCompletion * completion);

    /*
     * Obtain this object's provider.  We override the superclass's method
     * to return a more specific subclass of OSObject -- IOCDMedia.   This
     * method serves simply as a convenience to subclass developers.
     */

    virtual IOCDMedia * getProvider() const;

    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  0);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  1);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  2);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  3);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  4);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  5);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  6);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  7);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  8);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme,  9);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme, 10);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme, 11);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme, 12);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme, 13);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme, 14);
    OSMetaClassDeclareReservedUnused(IOCDPartitionScheme, 15);
};

#endif /* __cplusplus */
#endif /* KERNEL */
#endif /* !_IOCDPARTITIONSCHEME_H */
