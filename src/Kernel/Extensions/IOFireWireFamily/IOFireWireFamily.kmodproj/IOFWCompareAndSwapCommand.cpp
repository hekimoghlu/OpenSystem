/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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

OSDefineMetaClassAndStructors(IOFWCompareAndSwapCommand, IOFWAsyncCommand)
OSMetaClassDefineReservedUnused(IOFWCompareAndSwapCommand, 0);
OSMetaClassDefineReservedUnused(IOFWCompareAndSwapCommand, 1);
OSMetaClassDefineReservedUnused(IOFWCompareAndSwapCommand, 2);
OSMetaClassDefineReservedUnused(IOFWCompareAndSwapCommand, 3);

#pragma mark -

// initWithController
//
//

bool IOFWCompareAndSwapCommand::initWithController(IOFireWireController *control)
{
	bool result = true;
	
    result = IOFWAsyncCommand::initWithController(control);
						  
	// create member variables
	
	if( result )
	{
		result = createMemberVariables();
	}

	if( result )
	{
		result = createMemoryDescriptor();
	}
	
	return result;
}

// initAll
//
//

bool IOFWCompareAndSwapCommand::initAll(	IOFireWireNub *		device, 
											FWAddress 			devAddress,
											const UInt32 *		cmpVal, 
											const UInt32 *		newVal, 
											int 				size, 
											FWDeviceCallback	completion,
											void *				refcon, 
											bool 				failOnReset )
{
	bool result = true;
		
    result = IOFWAsyncCommand::initAll(	device, 
										devAddress,
										NULL, 
										completion, 
										refcon, 
										failOnReset );
	
	// create member variables
	
	if( result )
	{
		result = createMemberVariables();
	}
											
	if( result )
	{
		setInputVals( cmpVal, newVal, size );
	}
	
	if( result )
	{
		result = createMemoryDescriptor();
	}

	return result;
}

// initAll
//
//

bool IOFWCompareAndSwapCommand::initAll(	IOFireWireController *	control,
											UInt32 					generation, 
											FWAddress 				devAddress,
											const UInt32 *			cmpVal, 
											const UInt32 *			newVal, 
											int 					size,
											FWDeviceCallback 		completion, 
											void *					refcon )
{
    bool result = true;

	result = IOFWAsyncCommand::initAll(	control, 
										generation, 
										devAddress,
										NULL, 
										completion, 
										refcon );
	// create member variables
	
	if( result )
	{
		result = createMemberVariables();
	}
	
	if( result )
	{
		setInputVals( cmpVal, newVal, size );
	}

	if( result )
	{
		result = createMemoryDescriptor();
	}
	
	return result;
}

// createMemberVariables
//
//

bool IOFWCompareAndSwapCommand::createMemberVariables( void )
{
	bool success = true;
	fInputVals = NULL;
	
	if( fMembers == NULL )
	{
		// create member variables
		
		if( success )
		{
			fMembers = IOMallocType( MemberVariables );
			if( fMembers == NULL )
				success = false;
		}
		
		if( success )
		{
			// Using malloc here and not an IOBufferMemoryDescriptor as fMembers->fMemory
			// is used elsewhere and I don't want to mess with it
			fInputVals = (UInt32 *)IOMallocData( 4 * sizeof(UInt32) );
			if( NULL == fInputVals )	
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

void IOFWCompareAndSwapCommand::destroyMemberVariables( void )
{
	if( fMembers != NULL )
	{
		IOFreeType( fMembers, MemberVariables );
	}
	
	if( fInputVals != NULL )
	{
		IOFreeData( (void *)fInputVals, 4 * sizeof(UInt32) );
	}


}

// createMemoryDescriptor
//
//

bool IOFWCompareAndSwapCommand::createMemoryDescriptor()
{
	bool result = true;
	bool prepared = false;
	
	if( result )
	{
		fMembers->fMemory = IOMemoryDescriptor::withAddress( fInputVals, 4 * sizeof(UInt32), kIODirectionInOut );
        if( fMembers->fMemory == NULL )
            result = false;
	}
	
	if( result )
	{
		IOReturn status = fMembers->fMemory->prepare( kIODirectionInOut );
		if( status == kIOReturnSuccess )
		{
			prepared = true;
		}
		else
		{
			result = false;
		}
	}

	if( !result )
	{
		if( fMembers->fMemory != NULL )
		{
			if( prepared )
			{
				fMembers->fMemory->complete( kIODirectionInOut );
			}

			fMembers->fMemory->release();
			fMembers->fMemory = NULL;
		}
	}
		
	return result;
}

// destroyMemoryDescriptor
//
//

void IOFWCompareAndSwapCommand::destroyMemoryDescriptor()
{
	if( fMembers->fMemory != NULL )
	{
		fMembers->fMemory->complete( kIODirectionInOut );
		fMembers->fMemory->release();
		fMembers->fMemory = NULL;
	}
}
	
// free
//
//

void IOFWCompareAndSwapCommand::free()
{
	destroyMemoryDescriptor();

	destroyMemberVariables();

    IOFWAsyncCommand::free();
}

// reinit
//
//

IOReturn IOFWCompareAndSwapCommand::reinit(	FWAddress 			devAddress,
											const UInt32 *		cmpVal, 
											const UInt32 *		newVal, 
											int 				size,
											FWDeviceCallback	completion, 
											void *				refcon, 
											bool 				failOnReset )
{
    IOReturn status = kIOReturnSuccess;
    
	status = IOFWAsyncCommand::reinit( devAddress, NULL, completion, refcon, failOnReset );
    
	if( status == kIOReturnSuccess )
	{
		setInputVals( cmpVal, newVal, size );
	}
    
	return status;
}

// reinit
//
//

IOReturn IOFWCompareAndSwapCommand::reinit(	UInt32 				generation, 
											FWAddress 			devAddress,
											const UInt32 *		cmpVal, 
											const UInt32 *		newVal, 
											int 				size,
											FWDeviceCallback	completion, 
											void *				refcon )
{
	IOReturn status = kIOReturnSuccess;
    
    status = IOFWAsyncCommand::reinit( generation, devAddress, NULL, completion, refcon );
    
	if( status == kIOReturnSuccess )
	{
		setInputVals( cmpVal, newVal, size );
	}
	
    return status;
}

// setInputVals
//
//

void IOFWCompareAndSwapCommand::setInputVals( const UInt32 * cmpVal, const UInt32 * newVal, int size )
{
	int i;
	
    for( i = 0; i < size; i++ ) 
	{
        fInputVals[i] = cmpVal[i];
        fInputVals[size+i] = newVal[i];
    }

    fSize = 8 * size;
}

// execute
//
//

IOReturn IOFWCompareAndSwapCommand::execute()
{
    IOReturn result;
    fStatus = kIOReturnBusy;

	if( !fFailOnReset ) 
	{
        // Update nodeID and generation
        fDevice->getNodeIDGeneration(fGeneration, fNodeID);
		fSpeed = fControl->FWSpeed( fNodeID );
		if( IOFWAsyncCommand::fMembers->fMaxSpeed < fSpeed )
		{
			fSpeed = IOFWAsyncCommand::fMembers->fMaxSpeed;
		}
    }

    // Do this when we're in execute, not before,
    // so that Reset handling knows which commands are waiting a response.
    fTrans = fControl->allocTrans(this);
    if( fTrans ) 
	{
		result = fControl->asyncLock(	fGeneration, 
										fNodeID, 
										fAddressHi, 
										fAddressLo, 
										fSpeed,
										fTrans->fTCode, 
										kFWExtendedTCodeCompareSwap,
										fMembers->fMemory,
										0, 
										fSize, 
										this );
    }
    else
	{
//        IOLog("IOFWCompareAndSwapCommand::execute: Out of tLabels?\n");
        result = kIOFireWireOutOfTLabels;
    }

	// complete could release us so protect fStatus with retain and release
	IOReturn status = fStatus;	
    if( result != kIOReturnSuccess )
	{
		retain();
        complete(result);
		status = fStatus;
		release();
	}
	else
	{
		fBytesTransferred = fSize ;
	}
	
	return status;
}

// locked
//
//

bool IOFWCompareAndSwapCommand::locked( UInt32 * oldVal )
{
    int i;

    bool result = true;
    
	for( i = 0; i < fSize / 8; i++ ) 
	{
        result = result && (fInputVals[i] == fOldVal[i]);
		oldVal[i] = fOldVal[i];
    }
	
    return result;
}

// gotPacket
//
//

void IOFWCompareAndSwapCommand::gotPacket( int rcode, const void* data, int size )
{
	setResponseCode( rcode );
    IOReturn result = kIOReturnSuccess;
	
    unsigned int i;
    
	if( rcode != kFWResponseComplete ) 
	{
        //IOLog("Received rcode %d for lock command %p, nodeID %x\n", rcode, this, fNodeID);
        complete( kIOFireWireResponseBase + rcode );
        return;
    }
	
	if( size != (fSize / 2) )
	{
        // IOLog("Bad Lock Response Length for node %x (%08x instead of %08x)\n", fNodeID,size,fSize / 2);
		result = kIOFireWireInvalidResponseLength;
	}
	
	unsigned int clipped_size = size;
	if( clipped_size > sizeof(fOldVal) )
	{
		clipped_size = sizeof(fOldVal);
	}
	
    for( i = 0; i < clipped_size / 4; i++ ) 
	{
        fOldVal[i] = ((UInt32 *)data)[i];
    }
	
    complete( result );
}
