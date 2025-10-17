/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
 *
 *	IOFireWireUnit.h
 *
 *
 */
#ifndef _IOKIT_IOFIREWIREUNIT_H
#define _IOKIT_IOFIREWIREUNIT_H

// public
#include <IOKit/firewire/IOFireWireNub.h>

class IOFireWireDevice;
class IOFireWireUnit;

#pragma mark -

/*! 
	@class IOFireWireUnitAux
*/

class IOFireWireUnitAux : public IOFireWireNubAux
{
    OSDeclareDefaultStructors(IOFireWireUnitAux)

	friend class IOFireWireUnit;
	
protected:
	
	/*! 
		@struct ExpansionData
		@discussion This structure will be used to expand the capablilties of the class in the future.
    */  
	  
    struct ExpansionData { };

	/*! 
		@var reserved
		Reserved for future use.  (Internal use only)  
	*/
    
	ExpansionData * reserved;

    virtual bool init( IOFireWireUnit * primary );
	virtual	void free() APPLE_KEXT_OVERRIDE;

	virtual bool isPhysicalAccessEnabled( void ) APPLE_KEXT_OVERRIDE;

	virtual IOFWSimpleContiguousPhysicalAddressSpace * createSimpleContiguousPhysicalAddressSpace( vm_size_t size, IODirection direction ) APPLE_KEXT_OVERRIDE;
		
    virtual IOFWSimplePhysicalAddressSpace * createSimplePhysicalAddressSpace( vm_size_t size, IODirection direction ) APPLE_KEXT_OVERRIDE;
	
private:
    OSMetaClassDeclareReservedUnused(IOFireWireUnitAux, 0);
    OSMetaClassDeclareReservedUnused(IOFireWireUnitAux, 1);
    OSMetaClassDeclareReservedUnused(IOFireWireUnitAux, 2);
    OSMetaClassDeclareReservedUnused(IOFireWireUnitAux, 3);
	
};

#pragma mark -

/*! @class IOFireWireUnit
*/
class IOFireWireUnit : public IOFireWireNub
{
    OSDeclareDefaultStructors(IOFireWireUnit)

	friend class IOFireWireUnitAux;
	friend class IOFireWireDevice;

protected:
    IOFireWireDevice *fDevice;	// The device unit is part of

/*! @struct ExpansionData
    @discussion This structure will be used to expand the capablilties of the class in the future.
    */    
    struct ExpansionData { };

/*! @var reserved
    Reserved for future use.  (Internal use only)  */
    ExpansionData *reserved;

/*------------------Methods provided to FireWire device clients-----------------------*/
public:

    virtual bool init(OSDictionary *propTable, IOConfigDirectory *directory);
    
    /*
     * Standard nub initialization
     */
    virtual bool attach(IOService * provider ) APPLE_KEXT_OVERRIDE;
	virtual void free() APPLE_KEXT_OVERRIDE;

    /*
     * Matching language support
     * Match on the following properties of the unit:
     * Vendor_ID
     * GUID
     * Unit_Spec_ID
     * Unit_SW_Version
     */
    virtual bool matchPropertyTable(OSDictionary * table) APPLE_KEXT_OVERRIDE;


    virtual IOReturn message( UInt32 type, IOService * provider, void * argument ) APPLE_KEXT_OVERRIDE;

    // Override handleOpen() and handleClose() to pass on to device
    virtual bool handleOpen( 	IOService *	  forClient,
                                IOOptionBits	  options,
                                void *		  arg ) APPLE_KEXT_OVERRIDE;

    virtual void handleClose(   IOService *	  forClient,
                                IOOptionBits	  options ) APPLE_KEXT_OVERRIDE;
    
    virtual void setNodeFlags( UInt32 flags ) APPLE_KEXT_OVERRIDE;
	virtual void clearNodeFlags( UInt32 flags ) APPLE_KEXT_OVERRIDE;
    virtual UInt32 getNodeFlags( void ) APPLE_KEXT_OVERRIDE;

	virtual IOReturn setConfigDirectory( IOConfigDirectory *directory ) APPLE_KEXT_OVERRIDE;

    /*
     * Create local FireWire address spaces for the device to access
     */
    virtual IOFWPhysicalAddressSpace *createPhysicalAddressSpace(IOMemoryDescriptor *mem) APPLE_KEXT_OVERRIDE;
    virtual IOFWPseudoAddressSpace *createPseudoAddressSpace(FWAddress *addr, UInt32 len,
                    FWReadCallback reader, FWWriteCallback writer, void *refcon) APPLE_KEXT_OVERRIDE;

protected:
	
	virtual IOFireWireNubAux * createAuxiliary( void ) APPLE_KEXT_OVERRIDE;

public:
	void setMaxSpeed( IOFWSpeed speed );

protected:
	void terminateUnit( void );
	static void terminateUnitThreadFunc( void * refcon );
	    
private:
    OSMetaClassDeclareReservedUnused(IOFireWireUnit, 0);
    OSMetaClassDeclareReservedUnused(IOFireWireUnit, 1);

};

#endif /* ! _IOKIT_IOFIREWIREDEVICE_H */
