/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 5, 2022.
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

OSDefineMetaClassAndStructors(IOFWWriteQuadCommand, IOFWAsyncCommand)
OSMetaClassDefineReservedUnused(IOFWWriteQuadCommand, 0);
OSMetaClassDefineReservedUnused(IOFWWriteQuadCommand, 1);

#pragma mark -

// initWithController
//
//

bool IOFWWriteQuadCommand::initWithController(IOFireWireController *control)
{
	bool success = true;
	
    fWrite = true;
	
    success = IOFWAsyncCommand::initWithController(control);
						  
	// create member variables
	
	if( success )
	{
		success = createMemberVariables();
	}

	if( success )
	{
		success = createMemoryDescriptor();
	}
		
	return success;
}

// initAll
//
//

bool IOFWWriteQuadCommand::initAll(	IOFireWireNub *		device, 
									FWAddress 			devAddress,
									UInt32 *			quads, 
									int 				numQuads, 
									FWDeviceCallback	completion,
									void *				refcon, 
									bool 				failOnReset )
{
    bool	result = true;
	
	if( numQuads > kMaxWriteQuads )
	{
		result = false;
    }

	if( result )
	{
		
		fQuads = (UInt32 *)IOMallocData( kMaxWriteQuads * 4 );
		if( NULL == fQuads )
			result = false;
	}
		
	if( result )
	{
		
		fWrite = true;
		result = IOFWAsyncCommand::initAll(	device, 
											devAddress,
											NULL, 
											completion, 
											refcon, 
											failOnReset);
	}
	
	// create member variables
	
	if( result )
	{
		result = createMemberVariables();
	}
	
	if( result )
	{
		((MemberVariables*)fMembers->fSubclassMembers)->fDeferredNotify = false;
		setQuads( quads, numQuads );
	}

	if( result )
	{
		result = createMemoryDescriptor();
	}
		
	if( !result )
	{
		if( fQuads )
		{
			IOFreeData( fQuads, kMaxWriteQuads * 4 );
			fQuads = NULL;
		}
	}

	return result;
}

// initAll
//
//

bool IOFWWriteQuadCommand::initAll(	IOFireWireController *	control,
									UInt32 					generation, 
									FWAddress 				devAddress,
									UInt32 *				quads, 
									int 					numQuads, 
									FWDeviceCallback 		completion, 
									void *					refcon )
{
	bool 	result = true;
	
    if( numQuads > kMaxWriteQuads )
    {
	   result = false;
    }
	
	if( result )
	{
		
		fQuads = (UInt32 *)IOMallocData( kMaxWriteQuads * 4 );
		if( NULL == fQuads )
			result = false;
	}
		
	if( result )
	{
		fWrite = true;
		result = IOFWAsyncCommand::initAll(	control, 
											generation, 
											devAddress,
											NULL, 
											completion, 
											refcon );
	}

	// create member variables
	
	if( result )
	{
		result = createMemberVariables();
	}
		
	if( result )
	{
		((MemberVariables*)fMembers->fSubclassMembers)->fDeferredNotify = false;
		setQuads( quads, numQuads );
    }

	if( result )
	{
		result = createMemoryDescriptor();
	}
	
	if( !result )
	{
		if( fQuads )
		{
			IOFreeData( fQuads, kMaxWriteQuads * 4 );
			fQuads = NULL;
		}
	}
	return result;
}

// createMemberVariables
//
//

bool IOFWWriteQuadCommand::createMemberVariables( void )
{
	bool success = true;
	
	if( fMembers == NULL )
	{
		success = IOFWAsyncCommand::createMemberVariables();
	}
	
	if( fMembers && fMembers->fSubclassMembers == NULL )
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

void IOFWWriteQuadCommand::destroyMemberVariables( void )
{
	if( fMembers->fSubclassMembers != NULL )
	{		
		// free member variables
		
		IOFreeType( fMembers->fSubclassMembers, MemberVariables );
		fMembers->fSubclassMembers = NULL;
	}
}

// createMemoryDescriptor
//
//

bool IOFWWriteQuadCommand::createMemoryDescriptor( void )
{
	bool result = true;
	bool prepared = false;
	
	if( result )
	{
		((MemberVariables*)fMembers->fSubclassMembers)->fMemory = IOMemoryDescriptor::withAddress( fQuads, kMaxWriteQuads * sizeof(UInt32), kIODirectionInOut );
        if( ((MemberVariables*)fMembers->fSubclassMembers)->fMemory == NULL )
            result = false;
	}
	
	if( result )
	{
		IOReturn status = ((MemberVariables*)fMembers->fSubclassMembers)->fMemory->prepare( kIODirectionInOut );
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
		if( ((MemberVariables*)fMembers->fSubclassMembers)->fMemory != NULL )
		{
			if( prepared )
			{
				((MemberVariables*)fMembers->fSubclassMembers)->fMemory->complete( kIODirectionInOut );
			}

			((MemberVariables*)fMembers->fSubclassMembers)->fMemory->release();
			((MemberVariables*)fMembers->fSubclassMembers)->fMemory = NULL;
		}
	}
		
	return result;
}

// destroyMemoryDescriptor
//
//

void IOFWWriteQuadCommand::destroyMemoryDescriptor()
{
	if( (fMembers != NULL) &&
		(fMembers->fSubclassMembers != NULL) &&
		((MemberVariables*)fMembers->fSubclassMembers)->fMemory != NULL )
	{
		((MemberVariables*)fMembers->fSubclassMembers)->fMemory->complete( kIODirectionInOut );
		((MemberVariables*)fMembers->fSubclassMembers)->fMemory->release();
		((MemberVariables*)fMembers->fSubclassMembers)->fMemory = NULL;
	}
}
	
// free
//
//

void IOFWWriteQuadCommand::free()
{
	if( fQuads )
	{
		IOFreeData( fQuads, kMaxWriteQuads * 4 );
		fQuads = NULL;
	}
		
	destroyMemoryDescriptor();

	destroyMemberVariables();
	
    IOFWAsyncCommand::free();
}

// reinit
//
//

IOReturn IOFWWriteQuadCommand::reinit(	FWAddress 			devAddress,
										UInt32 *			quads, 
										int 				numQuads, 
										FWDeviceCallback	completion,
										void *				refcon, 
										bool 				failOnReset )
{
    IOReturn status = kIOReturnSuccess;
	
    if(numQuads > kMaxWriteQuads)
	{
		status = kIOReturnUnsupported;
    }
	
	if( status == kIOReturnSuccess )
	{
		status = IOFWAsyncCommand::reinit(	devAddress,
											NULL, 
											completion, 
											refcon, 
											failOnReset );
	}
	
    if( status == kIOReturnSuccess )
	{
		setQuads( quads, numQuads );
	}
	
    return status;
}

// reinit
//
//

IOReturn IOFWWriteQuadCommand::reinit(	UInt32 				generation, 
										FWAddress 			devAddress,
										UInt32 *			quads, 
										int 				numQuads, 
										FWDeviceCallback	completion, 
										void *				refcon )
{
    IOReturn status = kIOReturnSuccess;
    
	if( numQuads > kMaxWriteQuads )
	{
        status = kIOReturnUnsupported;
    }
	
	if( status == kIOReturnSuccess )
	{
		status = IOFWAsyncCommand::reinit( generation, devAddress, NULL, completion, refcon );
    }
	
	if( status == kIOReturnSuccess )
	{
		setQuads( quads, numQuads );
	}
	
	return status;
}

// setQuads
//
//

void IOFWWriteQuadCommand::setQuads( UInt32 * quads, int numQuads )
{
	int i;
	
	fSize = 4 * numQuads;
	
	for( i = 0; i < numQuads; i++ )
	{
		fQuads[i] = *quads++;
	}
	
	fQPtr = fQuads;
}

// execute
//
//

IOReturn IOFWWriteQuadCommand::execute()
{
    IOReturn result;

//	IOLog( "IOFWWriteQuadCommand::execute\n" );		
    
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
			
			if( ((IOFWAsyncCommand::MemberVariables*)fMembers)->fForceBlockRequests )
			{
				flags |= kIOFWWriteBlockRequest;
			}
		}
				
// IOLog( "IOFWWriteQuadCommand::execute - fControl->asyncWrite()\n" );		
        result = fControl->asyncWrite(	fGeneration, 
										fNodeID, 
										fAddressHi, 
										fAddressLo,
										fSpeed, 
										fTrans->fTCode, 
										((MemberVariables*)fMembers->fSubclassMembers)->fMemory,
										0,
										fPackSize, 
										this,
										(IOFWWriteFlags)flags );
    }
    else 
	{
//        IOLog("IOFWReadCommand::execute: Out of tLabels?\n");
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

void IOFWWriteQuadCommand::gotPacket( int rcode, const void* data, int size )
{
	setResponseCode( rcode );
	
    if( rcode != kFWResponseComplete ) 
	{
        //kprintf("Received rcode %d for command 0x%x\n", rcode, this);
        complete( kIOFireWireResponseBase+rcode );
        return;
    }
    else 
	{
        fQPtr += fPackSize / 4;
        fSize -= fPackSize;
		fBytesTransferred += fPackSize ;
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
