/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 4, 2024.
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
 * IOCDBlockStorageDriver.h
 *
 * This class implements  CD functionality, independent of
 * the physical connection protocol (e.g. SCSI, ATA, USB).
 *
 * A protocol-specific provider implements the functionality using an appropriate
 * protocol and commands.
 */

#ifndef	_IOCDBLOCKSTORAGEDRIVER_H
#define	_IOCDBLOCKSTORAGEDRIVER_H

#include <IOKit/IOTypes.h>
#include <IOKit/storage/IOCDBlockStorageDevice.h>
#include <IOKit/storage/IOCDTypes.h>
#include <IOKit/storage/IOBlockStorageDriver.h>

/*
 * @defined kIOCDBlockStorageDriverClass
 * @abstract
 * kIOCDBlockStorageDriverClass is the name of the IOCDBlockStorageDriver class.
 * @discussion
 * kIOCDBlockStorageDriverClass is the name of the IOCDBlockStorageDriver class.
 */

#define kIOCDBlockStorageDriverClass "IOCDBlockStorageDriver"

class IOCDBlockStorageDriver : public IOBlockStorageDriver {

    OSDeclareDefaultStructors(IOCDBlockStorageDriver)

public:

    static const UInt64 kBlockSizeCD = 2352;
    static const UInt8  kBlockTypeCD = 0x01;

    /* Overrides of IORegistryEntry */
    
    virtual bool	init(OSDictionary * properties);
    virtual void	free(void);

    /* Overrides of IOBlockStorageDriver: */

    virtual IOReturn	ejectMedia(void);
    virtual void 	executeRequest(UInt64 byteStart,
                	               IOMemoryDescriptor *buffer,
                	               IOStorageAttributes *attributes,
                	               IOStorageCompletion *completion,
                	               Context *context);
    virtual const char * getDeviceTypeName(void);
    virtual IOMedia *	instantiateDesiredMediaObject(void);
    virtual IOMedia *	instantiateMediaObject(UInt64 base,UInt64 byteSize,
                                            UInt32 blockSize,char *mediaName);
    virtual IOReturn	recordMediaParameters(void);
    
    /* End of IOBlockStorageDriver overrides. */

    /*
     * @function getMediaType
     * @abstract
     * Get the current type of media inserted in the CD drive.
     * @discussion
     * Certain I/O operations may not be allowed depending on the type of
     * media currently inserted. For example, one cannot issue write operations
     * if CD-ROM media is inserted.
     * @result
     * See the kCDMediaType constants in IOCDTypes.h.
     */
    virtual UInt32	getMediaType(void);

    /* -------------------------------------------------*/
    /* APIs implemented here, exported by IOCDMedia:    */
    /* -------------------------------------------------*/

    virtual CDTOC *	getTOC(void);
    virtual void	readCD(IOService *client,
                	       UInt64 byteStart,
                	       IOMemoryDescriptor *buffer,
                	       CDSectorArea sectorArea,
                	       CDSectorType sectorType,
                	       IOStorageAttributes *attributes,
                	       IOStorageCompletion *completion);
    virtual IOReturn	readISRC(UInt8 track,CDISRC isrc);
    virtual IOReturn	readMCN(CDMCN mcn);

    /* end of IOCDMedia APIs */
    
    /*
     * Obtain this object's provider.  We override the superclass's method to
     * return a more specific subclass of IOService -- IOCDBlockStorageDevice.  
     * This method serves simply as a convenience to subclass developers.
     */

    virtual IOCDBlockStorageDevice * getProvider() const;

protected:
        
    /* Overrides of IOBlockStorageDriver behavior. */

    /* When CD media is inserted, we want to create multiple nubs for the data and
     * audio tracks, for sessions, and the entire media. We override the methods
     * that manage nubs.
     */
    virtual IOReturn	acceptNewMedia(void);
    virtual IOReturn	decommissionMedia(bool forcible);

    /* End of IOBlockStorageDriver overrides. */

    /* Internally used methods: */

    using	IOBlockStorageDriver::getMediaBlockSize;

    virtual IOReturn	cacheTocInfo(void);
    virtual UInt64	getMediaBlockSize(CDSectorArea area,CDSectorType type);
    virtual void	prepareRequest(UInt64 byteStart,
                	               IOMemoryDescriptor *buffer,
                	               CDSectorArea sectorArea,
                	               CDSectorType sectorType,
                	               IOStorageAttributes *attributes,
                	               IOStorageCompletion *completion);

    /* ------- */

    struct ExpansionData
    {
        UInt32 minBlockNumberAudio;
        UInt32 maxBlockNumberAudio;
    };
    ExpansionData * _expansionData;

    #define _minBlockNumberAudio \
                IOCDBlockStorageDriver::_expansionData->minBlockNumberAudio
    #define _maxBlockNumberAudio \
                IOCDBlockStorageDriver::_expansionData->maxBlockNumberAudio

    UInt32				_reserved0032;

    /* We keep the TOC here because we'll always need it, so what the heck.
     *
     * There are possible "point" track entries for 0xa0..a2, 0xb0..b4, and 0xc0..0xc1.
     * Tracks need not start at 1, as long as they're between 1 and 99, and have contiguous
     * numbers.
     */

    CDTOC *				_toc;
    UInt32				_tocSize;

    /* ------- */

    IOReturn	reportDiscInfo(CDDiscInfo *discInfo);
    IOReturn	reportTrackInfo(UInt16 track,CDTrackInfo *trackInfo);

public:

    virtual IOReturn	getSpeed(UInt16 * kilobytesPerSecond);

    virtual IOReturn	setSpeed(UInt16 kilobytesPerSecond);

    virtual IOReturn	readTOC(IOMemoryDescriptor *buffer,CDTOCFormat format,
                    	        UInt8 formatAsTime,UInt8 trackOrSessionNumber,
                    	        UInt16 *actualByteCount);

    virtual IOReturn	readDiscInfo(IOMemoryDescriptor *buffer,
                    	             UInt16 *actualByteCount);

    virtual IOReturn	readTrackInfo(IOMemoryDescriptor *buffer,UInt32 address,
                    	              CDTrackInfoAddressType addressType,
                    	              UInt16 *actualByteCount);

    virtual void	writeCD(IOService *client,
                	        UInt64 byteStart,
                	        IOMemoryDescriptor *buffer,
                	        CDSectorArea sectorArea,
                	        CDSectorType sectorType,
                	        IOStorageAttributes *attributes,
                	        IOStorageCompletion *completion);

    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  0);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  1);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  2);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  3);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  4);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  5);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  6);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  7);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  8);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver,  9);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver, 10);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver, 11);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver, 12);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver, 13);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver, 14);
    OSMetaClassDeclareReservedUnused(IOCDBlockStorageDriver, 15);
};
#endif
