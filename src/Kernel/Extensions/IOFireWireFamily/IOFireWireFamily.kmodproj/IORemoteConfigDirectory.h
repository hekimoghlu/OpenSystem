/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#ifndef __IOREMOTECONFIGDIRECTORY_H__
#define __IOREMOTECONFIGDIRECTORY_H__

#include <libkern/c++/OSObject.h>
#include <IOKit/IOReturn.h>

#include <IOKit/firewire/IOFireWireFamilyCommon.h>
#include <IOKit/firewire/IOConfigDirectory.h>

#include "IOFireWireROMCache.h"

class OSString;
class OSIterator;
class IOFireWireDevice;

/*! @class IORemoteConfigDirectory
*/
class IORemoteConfigDirectory : public IOConfigDirectory
{
    OSDeclareDefaultStructors(IORemoteConfigDirectory);

protected:
    IOFireWireROMCache *fROM;				// Our cache of the ROM
    
/*! @struct ExpansionData
    @discussion This structure will be used to expand the capablilties of the class in the future.
    */    
    struct ExpansionData { };

/*! @var reserved
    Reserved for future use.  (Internal use only)  */
    ExpansionData *reserved;

    virtual bool initWithOwnerOffset(IOFireWireROMCache *rom,
                             int start, int type);
    virtual void free(void) APPLE_KEXT_OVERRIDE;

    virtual const UInt32 *getBase(void) APPLE_KEXT_OVERRIDE;
    virtual IOConfigDirectory *getSubDir(int start, int type) APPLE_KEXT_OVERRIDE;

public:
    static IOConfigDirectory *withOwnerOffset(IOFireWireROMCache *rom,
                                           int start, int type);


    /*!
        @function update
        makes sure that the ROM has at least the specified capacity,
        and that the ROM is uptodate from its start to at least the
        specified quadlet offset.
        @result kIOReturnSuccess if the specified offset is now
        accessable at romBase[offset].
    */
    virtual IOReturn update(UInt32 offset, const UInt32 *&romBase) APPLE_KEXT_OVERRIDE;

protected:
	
	virtual const UInt32 * lockData( void ) APPLE_KEXT_OVERRIDE;
	virtual void unlockData( void ) APPLE_KEXT_OVERRIDE;
	virtual IOReturn updateROMCache( UInt32 offset, UInt32 length ) APPLE_KEXT_OVERRIDE;
	virtual IOReturn checkROMState( void ) APPLE_KEXT_OVERRIDE;
	
private:
    OSMetaClassDeclareReservedUnused(IORemoteConfigDirectory, 0);
    OSMetaClassDeclareReservedUnused(IORemoteConfigDirectory, 1);
    OSMetaClassDeclareReservedUnused(IORemoteConfigDirectory, 2);
};


#endif /* __IOREMOTECONFIGDIRECTORY_H__ */
