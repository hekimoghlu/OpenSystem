/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 26, 2025.
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

OSDefineMetaClassAndStructors(IOFWReadQuadCommand, IOFWAsyncCommand)
OSMetaClassDefineReservedUnused(IOFWReadQuadCommand, 0);
OSMetaClassDefineReservedUnused(IOFWReadQuadCommand, 1);

#pragma mark -

// gotPacket
//
//

void IOFWReadQuadCommand::gotPacket(int rcode, const void* data, int size)
{
	setResponseCode( rcode );

    if(rcode != kFWResponseComplete) {
		// Sony CRX1600XL responses address error to breads to ROM
        if( (rcode == kFWResponseTypeError || rcode == kFWResponseAddressError) && fMaxPack > 4) {
            // try reading a quad at a time
            fMaxPack = 4;
            size = 0;
        }
        else {
            complete(kIOFireWireResponseBase+rcode);
            return;
        }
    }
    else {
        int i;
        UInt32 *src = (UInt32 *)data;
        for(i=0; i<size/4; i++)
        {
			*fQuads++ = *src++;
        }
        fSize -= size;
		
		// nwg: should update bytes transferred count
		fBytesTransferred += size ;
    }

    if(fSize > 0) {
        fAddressLo += size;
        updateTimer();
        fCurRetries = fMaxRetries;
        fControl->freeTrans(fTrans);  // Free old tcode

        execute();
    }
    else {
        complete(kIOReturnSuccess);
    }
}

// initAll
//
//

bool IOFWReadQuadCommand::initAll(IOFireWireNub *device, FWAddress devAddress,
	UInt32 *quads, int numQuads, FWDeviceCallback completion,
	void *refcon, bool failOnReset)
{
	bool success = true;
	
    success = IOFWAsyncCommand::initAll(device, devAddress,
										NULL, completion, refcon, failOnReset);
	
	// create member variables
	
	if( success )
	{
		fSize = 4*numQuads;
		fQuads = quads;
		
		success = createMemberVariables();
	}
	
	return success;
}

// initAll
//
//

bool IOFWReadQuadCommand::initAll(IOFireWireController *control,
                                  UInt32 generation, FWAddress devAddress,
        UInt32 *quads, int numQuads, FWDeviceCallback completion,
        void *refcon)
{	bool success = true;
	
    success = IOFWAsyncCommand::initAll(control, generation, devAddress,
										NULL, completion, refcon);
	
	// create member variables
	
	if( success )
	{
		fSize = 4*numQuads;
		fQuads = quads;
		
		success = createMemberVariables();
	}
	
	return success;
}

// createMemberVariables
//
//

bool IOFWReadQuadCommand::createMemberVariables( void )
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

void IOFWReadQuadCommand::destroyMemberVariables( void )
{
	if( fMembers->fSubclassMembers != NULL )
	{		
		// free member variables
		
		IOFreeType( fMembers->fSubclassMembers, MemberVariables );
		fMembers->fSubclassMembers = NULL;
	}
}

// free
//
//

void IOFWReadQuadCommand::free()
{	
	destroyMemberVariables();
	
	IOFWAsyncCommand::free();
}


// reinit
//
//

IOReturn IOFWReadQuadCommand::reinit(FWAddress devAddress,
	UInt32 *quads, int numQuads, FWDeviceCallback completion,
					void *refcon, bool failOnReset)
{
    IOReturn res;
    res = IOFWAsyncCommand::reinit(devAddress,
	NULL, completion, refcon, failOnReset);
    if(res != kIOReturnSuccess)
	return res;

    fSize = 4*numQuads;
    fQuads = quads;
    return res;
}

// reinit
//
//

IOReturn IOFWReadQuadCommand::reinit(UInt32 generation, FWAddress devAddress,
        UInt32 *quads, int numQuads, FWDeviceCallback completion, void *refcon)
{
    IOReturn res;
    res = IOFWAsyncCommand::reinit(generation, devAddress,
        NULL, completion, refcon);
    if(res != kIOReturnSuccess)
        return res;

    fSize = 4*numQuads;
    fQuads = quads;
    return res;
}

// execute
//
//

IOReturn IOFWReadQuadCommand::execute()
{
    IOReturn result;
    int transfer;

    fStatus = kIOReturnBusy;
    
    if(!fFailOnReset) {
        // Update nodeID and generation
        fDevice->getNodeIDGeneration(fGeneration, fNodeID);
		fSpeed = fControl->FWSpeed( fNodeID );  
		if( fMembers->fMaxSpeed < fSpeed )
		{
			fSpeed = fMembers->fMaxSpeed;
		}  
    }

    transfer = fSize;
    if(transfer > fMaxPack)
	{	
		transfer = fMaxPack;
	}
	
	int maxPack = (1 << fControl->maxPackLog(fWrite, fNodeID));
	if( maxPack < transfer )
	{
		transfer = maxPack;
	}

	UInt32 flags = kIOFWReadFlagsNone;

	if( fMembers )
	{
		if( ((IOFWAsyncCommand::MemberVariables*)fMembers)->fForceBlockRequests )
		{
			flags |= kIOFWReadBlockRequest;
		}
		
		if( ((MemberVariables*)fMembers->fSubclassMembers)->fPingTime )
		{
			flags |= kIOFWReadPingTime;
		}
	}
	
    // Do this when we're in execute, not before,
    // so that Reset handling knows which commands are waiting a response.
    fTrans = fControl->allocTrans(this);
    if(fTrans) {
        result = fControl->asyncRead(fGeneration, fNodeID, fAddressHi,
                        fAddressLo, fSpeed, fTrans->fTCode, transfer, this, (IOFWReadFlags)flags );
    }
    else {
	//	IOLog("IOFWReadCommand::execute: Out of tLabels?\n");
        result = kIOFireWireOutOfTLabels;
    }

	// complete could release us so protect fStatus with retain and release
	IOReturn status = fStatus;	
    if(result != kIOReturnSuccess)
	{
		retain();
        complete(result);
		status = fStatus;
		release();
	}
	return status;
}
