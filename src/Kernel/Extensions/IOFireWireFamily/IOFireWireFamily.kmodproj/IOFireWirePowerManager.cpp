/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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
#include <IOKit/firewire/IOFireWirePowerManager.h>
// protected
#include <IOKit/firewire/IOFireWireController.h>

// private
#import "FWDebugging.h"

OSDefineMetaClassAndStructors(IOFireWirePowerManager, OSObject)
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 0);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 1);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 2);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 3);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 4);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 5);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 6);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 7);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 8);
OSMetaClassDefineReservedUnused(IOFireWirePowerManager, 9);

#pragma mark -

/////////////////////////////////////////////////////////////////////////////

// createWithController
//
//

IOFireWirePowerManager * IOFireWirePowerManager::createWithController( IOFireWireController * controller )
{
    IOFireWirePowerManager * me = OSTypeAlloc( IOFireWirePowerManager );
	if( me != NULL )
	{
		if( !me->initWithController(controller) ) 
		{
            me->release();
            me = NULL;
        }
	}

    return me;
}

// initWithController
//
//

bool IOFireWirePowerManager::initWithController( IOFireWireController * controller )
{
	bool success = true;		// assume success
	
	// init super
	
    if( !OSObject::init() )
        success = false;
	
	if( success )
	{
		fControl = controller;
		fMaximumDeciwatts = 0;
		fAllocatedDeciwatts = 0;
	}
	
	return success;
}

#pragma mark -

/////////////////////////////////////////////////////////////////////////////

// setMaximumDeciwatts
//
//

void IOFireWirePowerManager::setMaximumDeciwatts( UInt32 deciwatts )
{
	FWKLOG(( "IOFireWirePowerManager::setMaximumDeciwatts - setting maximum milliwats to %d\n", deciwatts ));
	
	fMaximumDeciwatts = deciwatts;
}

// allocateDeciwatts
//
//

IOReturn IOFireWirePowerManager::allocateDeciwatts( UInt32 deciwatts )
{
	IOReturn status = kIOReturnSuccess;
	
	fControl->closeGate();
	
	FWKLOG(( "IOFireWirePowerManager::allocateDeciwatts - allocating %d deciwatts\n", deciwatts ));
	
	if( fAllocatedDeciwatts + deciwatts <= fMaximumDeciwatts )
	{
		fAllocatedDeciwatts += deciwatts;
	}
	else
	{
		status = kIOReturnNoResources;
	}
	
	fControl->openGate();
	
	return status;
}

// deallocateDeciwatts
//
//

void IOFireWirePowerManager::deallocateDeciwatts( UInt32 deciwatts )
{
	fControl->closeGate();
	
	FWKLOG(( "IOFireWirePowerManager::deallocateDeciwatts - freeing %d deciwatts\n", deciwatts ));
	
	if( deciwatts <= fAllocatedDeciwatts )
	{
		fAllocatedDeciwatts -= deciwatts;
	}
	else
	{
		IOLog( "IOFireWirePowerManager::deallocateDeciwatts - freed deciwatts %d > allocated deciwatts %d!\n", (uint32_t)deciwatts, (uint32_t)fAllocatedDeciwatts );
		fAllocatedDeciwatts = 0;
	}
	
	// notify clients that more power has been made available
	if( deciwatts != 0 )
	{
		fControl->messageClients( kIOFWMessagePowerStateChanged );
	}

	fControl->openGate();
}
