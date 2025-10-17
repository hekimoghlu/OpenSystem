/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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
//#define IOASSERT 1	// Set to 1 to activate assert()

// public
#include <IOKit/firewire/IOFWCommand.h>
#include <IOKit/firewire/IOFireWireController.h>
#include <IOKit/firewire/IOFireWireNub.h>
#include <IOKit/firewire/IOLocalConfigDirectory.h>

// system
#include <IOKit/assert.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOCommand.h>

OSDefineMetaClassAndStructors(IOFWWriteCommand, IOFWAsyncCommand)
OSMetaClassDefineReservedUnused(IOFWWriteCommand, 0);
OSMetaClassDefineReservedUnused(IOFWWriteCommand, 1);

#pragma mark -

// initWithController
//
//

bool IOFWWriteCommand::initWithController(IOFireWireController *control)
{
	bool success = true;
	
    fWrite = true;
	
    success = IOFWAsyncCommand::initWithController(control);
						  
	// create member variables
	
	if( success )
	{
		success = createMemberVariables();
	}
	
	return success;
}

// initAll
//
//

bool IOFWWriteCommand::initAll(	IOFireWireNub *			device, 
								FWAddress 				devAddress,
								IOMemoryDescriptor *	hostMem, 
								FWDeviceCallback 		completion,
								void *					refcon, 
								bool 					failOnReset )
{
	bool success = true;
	
    fWrite = true;
	
    success = IOFWAsyncCommand::initAll( device, devAddress, hostMem, 
										 completion, refcon, failOnReset);
						  
	// create member variables
	
	if( success )
	{
		success = createMemberVariables();
	}
	
	return success;
}

// initAll
//
//

bool IOFWWriteCommand::initAll(	IOFireWireController *	control,
								UInt32 					generation, 
								FWAddress 				devAddress,
								IOMemoryDescriptor *	hostMem, 
								FWDeviceCallback 		completion, 
								void *					refcon )
{
	bool success = true;
	
    fWrite = true;

    success = IOFWAsyncCommand::initAll(control, generation, devAddress,
										hostMem, completion, refcon);
						  
	// create member variables
	
	if( success )
	{
		success = createMemberVariables();
	}
	
	return success;
}

// createMemberVariables
//
//

bool IOFWWriteCommand::createMemberVariables( void )
{
	bool success = true;
	
	if( fMembers == NULL )
	{
		success = IOFWAsyncCommand::createMemberVariables();
	}
	
	if( fMembers )
	{
		if( success )
		{
			fMembers->fSubclassMembers = IOMallocType( MemberVariables );
			if( fMembers->fSubclassMembers == NULL )
				success = false;
		}
		
		// clean up on failure
		
		if( !success )
		{
			destroyMemberVariables();
		}
	}
	
	return success;
}

// destroyMemberVariables
//
//

void IOFWWriteCommand::destroyMemberVariables( void )
{
	if( fMembers->fSubclassMembers != NULL )
	{		
		// free member variables
		
		IOFreeType( fMembers->fSubclassMembers, MemberVariables );
	}
}

// free
//
//

void IOFWWriteCommand::free()
{	
	destroyMemberVariables();
	
	IOFWAsyncCommand::free();
}

// reinit
//
//

IOReturn IOFWWriteCommand::reinit(	FWAddress 				devAddress,
									IOMemoryDescriptor *	hostMem,
									FWDeviceCallback 		completion, 
									void *					refcon, 
									bool failOnReset )
{
    return IOFWAsyncCommand::reinit(	devAddress,
										hostMem, 
										completion, 
										refcon, 
										failOnReset );
}

// reinit
//
//

IOReturn IOFWWriteCommand::reinit(	UInt32 					generation, 
									FWAddress 				devAddress,
									IOMemoryDescriptor *	hostMem,
									FWDeviceCallback 		completion, 
									void *					refcon )
{
    return IOFWAsyncCommand::reinit(	generation, 
										devAddress,
										hostMem, 
										completion, 
										refcon );
}

// execute
//
//

IOReturn IOFWWriteCommand::execute()
{
    IOReturn result;
    fStatus = kIOReturnBusy;

    if( !fFailOnReset ) 
	{
        // Update nodeID and generation
        fDevice->getNodeIDGeneration( fGeneration, fNodeID );
		fSpeed = fControl->FWSpeed( fNodeID );
		if( fMembers->fMaxSpeed < fSpeed )
		{
			fSpeed = fMembers->fMaxSpeed;
		}    
    }

    fPackSize = fSize;
    if( fPackSize > fMaxPack )
	{
		fPackSize = fMaxPack;
	}
	
	int maxPack = (1 << fControl->maxPackLog(fWrite, fNodeID));
	if( maxPack < fPackSize )
	{
		fPackSize = maxPack;
	}

    // Do this when we're in execute, not before,
    // so that Reset handling knows which commands are waiting a response.
    fTrans = fControl->allocTrans( this );
    if( fTrans ) 
	{
		UInt32 flags = kIOFWWriteFlagsNone;
		
		if( fMembers && fMembers->fSubclassMembers )
		{
			if( ((MemberVariables*)fMembers->fSubclassMembers)->fDeferredNotify )
			{
				flags |= kIOFWWriteFlagsDeferredNotify;
			}
			
			if( ((MemberVariables*)fMembers->fSubclassMembers)->fFastRetryOnBusy )
			{
				flags |= kIOFWWriteFastRetryOnBusy;
			}

			if( ((IOFWAsyncCommand::MemberVariables*)fMembers)->fForceBlockRequests )
			{
				flags |= kIOFWWriteBlockRequest;
			}
		}
		
        result = fControl->asyncWrite(	fGeneration, 
										fNodeID, 
										fAddressHi, 
										fAddressLo, 
										fSpeed,
										fTrans->fTCode, 
										fMemDesc, 
										fBytesTransferred, 
										fPackSize, 
										this,
										(IOFWWriteFlags)flags );
    }
    else 
	{
		//IOLog("IOFWWriteCommand::execute: Out of tLabels?\n");
        result = kIOFireWireOutOfTLabels;
    }

	// complete could release us so protect fStatus with retain and release
	IOReturn status = fStatus;	
    if( result != kIOReturnSuccess )
	{
		retain();
        complete( result );
		status = fStatus;
		release();
	}
	
	return status;
}

// gotPacket
//
//

void IOFWWriteCommand::gotPacket( int rcode, const void* data, int size )
{
	setResponseCode( rcode );
	
    if( rcode != kFWResponseComplete ) 
	{
        complete( kIOFireWireResponseBase+rcode );
        return;
    }
    else 
	{
		fBytesTransferred += fPackSize;
        fSize -= fPackSize;
    }

    if( fSize > 0 ) 
	{
        fAddressLo += fPackSize;

        updateTimer();
        fCurRetries = fMaxRetries;
        fControl->freeTrans( fTrans );  // Free old tcode
        execute();
    }
    else 
	{
        complete( kIOReturnSuccess );
    }
}
