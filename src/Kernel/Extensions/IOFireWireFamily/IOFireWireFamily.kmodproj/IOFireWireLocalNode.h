/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
	$Log: not supported by cvs2svn $
	Revision 1.7  2008/11/14 00:17:12  arulchan
	fix for rdar://5939334
	
	Revision 1.6  2005/02/18 22:56:53  gecko1
	3958781 Q45C EVT: FireWire ASP reporter says port speed is 800 Mb/sec
	
	Revision 1.5  2003/02/20 02:00:12  collin
	*** empty log message ***
	
	Revision 1.4  2003/02/17 21:47:53  collin
	*** empty log message ***
	
	Revision 1.3  2002/10/18 23:29:44  collin
	fix includes, fix cast which fails on new compiler
	
	Revision 1.2  2002/09/25 00:27:24  niels
	flip your world upside-down
	
*/

// public
#import <IOKit/firewire/IOFireWireNub.h>

class IOFireWireLocalNode;

#pragma mark -

/*! 
	@class IOFireWireLocalNodeAux
*/

class IOFireWireLocalNodeAux : public IOFireWireNubAux
{
    OSDeclareDefaultStructors(IOFireWireLocalNodeAux)

	friend class IOFireWireLocalNode;
	
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

    virtual bool init( IOFireWireLocalNode * primary );
	virtual	void free(void) APPLE_KEXT_OVERRIDE;
	
private:
    OSMetaClassDeclareReservedUnused(IOFireWireLocalNodeAux, 0);
    OSMetaClassDeclareReservedUnused(IOFireWireLocalNodeAux, 1);
    OSMetaClassDeclareReservedUnused(IOFireWireLocalNodeAux, 2);
    OSMetaClassDeclareReservedUnused(IOFireWireLocalNodeAux, 3);
	
};

#pragma mark -

/*! @class IOFireWireLocalNode
*/

class IOFireWireLocalNode : public IOFireWireNub
{
    OSDeclareDefaultStructors(IOFireWireLocalNode)

	friend class IOFireWireLocalNodeAux;

	/*------------------Useful info about device (also available in the registry)--------*/
protected:

	/*-----------Methods provided to FireWire device clients-------------*/
public:
	
		// Set up properties affected by bus reset
		virtual void setNodeProperties(UInt32 generation, UInt16 nodeID, UInt32 *selfIDs, int numIDs, IOFWSpeed maxSpeed );
		
		/*
		* Standard nub initialization
		*/
		virtual bool init(OSDictionary * propTable) APPLE_KEXT_OVERRIDE;
		virtual bool attach(IOService * provider ) APPLE_KEXT_OVERRIDE;
	
		virtual void handleClose(   IOService *	  forClient,
								IOOptionBits	  options ) APPLE_KEXT_OVERRIDE;
		virtual bool handleOpen( 	IOService *	  forClient,
								IOOptionBits	  options,
								void *		  arg ) APPLE_KEXT_OVERRIDE;
		virtual bool handleIsOpen(  const IOService * forClient ) const APPLE_KEXT_OVERRIDE;
	
		/*
		* Trick method to create protocol user clients
		*/
		virtual IOReturn setProperties( OSObject * properties ) APPLE_KEXT_OVERRIDE;

protected:
	
	virtual IOFireWireNubAux * createAuxiliary( void ) APPLE_KEXT_OVERRIDE;

public:
	virtual IOReturn message( UInt32 type, IOService * provider, void * argument ) APPLE_KEXT_OVERRIDE;
	virtual void free(void) APPLE_KEXT_OVERRIDE;

protected:
	OSSet * fOpenClients;
};
