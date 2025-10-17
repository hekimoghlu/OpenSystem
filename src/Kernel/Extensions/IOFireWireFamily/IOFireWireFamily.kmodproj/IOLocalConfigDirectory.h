/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#ifndef __IOLOCALCONFIGDIRECTORY_H__
#define __IOLOCALCONFIGDIRECTORY_H__

#include <libkern/c++/OSObject.h>
#include <IOKit/IOReturn.h>
#include <IOKit/firewire/IOFireWireFamilyCommon.h>
#include <IOKit/firewire/IOConfigDirectory.h>

class OSArray;
class OSData;
class IOFireWireController;
class IOFWUserObjectExporter ;

/*! @class IOLocalConfigDirectory
*/
class IOLocalConfigDirectory : public IOConfigDirectory
{
	friend class IOFireWireController;
	friend class IOFireWireUserClient ;

	OSDeclareDefaultStructors(IOLocalConfigDirectory);

protected:
	OSArray *fEntries;	// Entries for this directory.
	OSData *fROM;	// Local ROM, if compiled.
	UInt32 fHeader;	// Num entries and CRC.
	
/*! @struct ExpansionData
	@discussion This structure will be used to expand the capablilties of the class in the future.
	*/    
	struct ExpansionData { };

/*! @var reserved
	Reserved for future use.  (Internal use only)  */
	ExpansionData *reserved;

	virtual bool init(void) APPLE_KEXT_OVERRIDE;
	virtual void free(void) APPLE_KEXT_OVERRIDE;

	virtual const UInt32 *getBase(void) APPLE_KEXT_OVERRIDE;
	virtual IOConfigDirectory *getSubDir(int start, int type) APPLE_KEXT_OVERRIDE;

public:
	static IOLocalConfigDirectory *create();

	/*!
		@function update
		makes sure that the ROM has at least the specified capacity,
		and that the ROM is uptodate from its start to at least the
		specified quadlet offset.
		@result kIOReturnSuccess if the specified offset is now
		accessable at romBase[offset].
	*/
	virtual IOReturn update(UInt32 offset, const UInt32 *&romBase) APPLE_KEXT_OVERRIDE;

	virtual IOReturn compile(OSData *rom);
	
	// All flavours of addEntry eat a retain of the desc string
	virtual IOReturn addEntry(int key, UInt32 value, OSString *desc = NULL);
	virtual IOReturn addEntry(int key, IOLocalConfigDirectory *value,
							OSString *desc = NULL);
	virtual IOReturn addEntry(int key, OSData *value, OSString *desc = NULL);
	virtual IOReturn addEntry(int key, FWAddress value, OSString *desc = NULL);
	virtual IOReturn removeSubDir(IOLocalConfigDirectory *value);
	const OSArray *getEntries() const;

	virtual IOReturn getIndexValue(int index, IOConfigDirectory *&value) APPLE_KEXT_OVERRIDE;

protected:

	virtual const UInt32 * lockData( void ) APPLE_KEXT_OVERRIDE;
	virtual void unlockData( void ) APPLE_KEXT_OVERRIDE;
	virtual IOReturn updateROMCache( UInt32 offset, UInt32 length ) APPLE_KEXT_OVERRIDE;
	virtual IOReturn checkROMState( void ) APPLE_KEXT_OVERRIDE;

	// call eats a retain count
	virtual IOReturn addEntry(OSString *desc);

	IOReturn	incrementGeneration( void );
	static void	exporterCleanup( const OSObject * self, IOFWUserObjectExporter * exporter ) ;
		
private:
	OSMetaClassDeclareReservedUsed(IOLocalConfigDirectory, 0);
	OSMetaClassDeclareReservedUnused(IOLocalConfigDirectory, 1);
	OSMetaClassDeclareReservedUnused(IOLocalConfigDirectory, 2);
};

#endif /* __IOLOCALCONFIGDIRECTORY_H__ */
