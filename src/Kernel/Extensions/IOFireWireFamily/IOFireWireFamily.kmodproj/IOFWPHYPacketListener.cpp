/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
// public
#include <IOKit/firewire/IOFireWireController.h>
#include <IOKit/firewire/IOFWPHYPacketListener.h>

#include "FWDebugging.h"

OSDefineMetaClassAndStructors( IOFWPHYPacketListener, OSObject )

OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 0 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 1 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 2 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 3 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 4 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 5 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 6 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 7 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 8 );
OSMetaClassDefineReservedUnused( IOFWPHYPacketListener, 9 );

// createWithController
//
//

IOFWPHYPacketListener * IOFWPHYPacketListener::createWithController( IOFireWireController * controller )
{
    IOReturn				status = kIOReturnSuccess;
    IOFWPHYPacketListener * me = NULL;
        
    if( status == kIOReturnSuccess )
    {
        me = OSTypeAlloc( IOFWPHYPacketListener );
        if( me == NULL )
            status = kIOReturnNoMemory;
    }
    
    if( status == kIOReturnSuccess )
    {
        bool success = me->initWithController( controller );
		if( !success )
		{
			status = kIOReturnError;
		}
    }
    
    if( status != kIOReturnSuccess )
    {
        me = NULL;
    }

	FWKLOG(( "IOFWPHYPacketListener::create() - created new IOFWPHYPacketListener %p\n", me ));
    
    return me;
}

// initWithController
//
//

bool IOFWPHYPacketListener::initWithController( IOFireWireController * control )
{	
	bool success = OSObject::init();
	FWPANICASSERT( success == true );
	
	fControl = control;

	FWKLOG(( "IOFWPHYPacketListener::initWithController() - IOFWPHYPacketListener %p initialized\n", this  ));

	return true;
}

// free
//
//

void IOFWPHYPacketListener::free()
{	
	FWKLOG(( "IOFWPHYPacketListener::free() - freeing IOFWPHYPacketListener %p\n", this ));

	OSObject::free();
}

///////////////////////////////////////////////////////////////////////////////////////
#pragma mark -

// activate
//
//

IOReturn IOFWPHYPacketListener::activate( void )
{
    return fControl->activatePHYPacketListener( this );
}

// deactivate
//
//

void IOFWPHYPacketListener::deactivate( void )
{
    fControl->deactivatePHYPacketListener( this );
}

// processPHYPacket
//
//

void IOFWPHYPacketListener::processPHYPacket( UInt32 data1, UInt32 data2 )
{
	IOLog( "IOFWPHYPacketListener<%p>::processPHYPacket - 0x%x 0x%x\n", this, (uint32_t)data1, (uint32_t)data2 );
	
	if( fCallback )
	{
		(fCallback)( fRefCon, data1, data2 );
	}
}

// setCallback
//
//

void IOFWPHYPacketListener::setCallback( FWPHYPacketCallback callback )
{
	fCallback = callback;
}

// setRefCon
//
//

void IOFWPHYPacketListener::setRefCon( void * refcon )
{
	fRefCon = refcon;
}

// getRefCon
//
//

void * IOFWPHYPacketListener::getRefCon( void )
{
	return fRefCon;
}
