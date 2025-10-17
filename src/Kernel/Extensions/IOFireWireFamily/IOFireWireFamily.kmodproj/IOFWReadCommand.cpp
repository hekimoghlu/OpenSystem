/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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

OSDefineMetaClassAndStructors(IOFWReadCommand, IOFWAsyncCommand)
OSMetaClassDefineReservedUnused(IOFWReadCommand, 0);
OSMetaClassDefineReservedUnused(IOFWReadCommand, 1);

#pragma mark -

// gotPacket
//
//

void IOFWReadCommand::gotPacket(int rcode, const void* data, int size)
{
	setResponseCode( rcode );
	
    if(rcode != kFWResponseComplete) {
        //kprintf("Received rcode %d for read command 0x%x, nodeID %x\n", rcode, this, fNodeID);
        if(rcode == kFWResponseTypeError && fMaxPack > 4) {
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
        fMemDesc->writeBytes(fBytesTransferred, data, size);
        fSize -= size;
	fBytesTransferred += size;
    }

    if(fSize > 0) {
        fAddressLo += size;
        fControl->freeTrans(fTrans);  // Free old tcode
        updateTimer();
        fCurRetries = fMaxRetries;
        execute();
    }
    else {
        complete(kIOReturnSuccess);
    }
}

// initAll
//
//

bool IOFWReadCommand::initAll(IOFireWireNub *device, FWAddress devAddress,
	IOMemoryDescriptor *hostMem, FWDeviceCallback completion,
	void *refcon, bool failOnReset)
{
    return IOFWAsyncCommand::initAll(device, devAddress,
                          hostMem, completion, refcon, failOnReset);
}

// initAll
//
//

bool IOFWReadCommand::initAll(IOFireWireController *control,
                              UInt32 generation, FWAddress devAddress,
        IOMemoryDescriptor *hostMem, FWDeviceCallback completion,
        void *refcon)
{
    return IOFWAsyncCommand::initAll(control, generation, devAddress,
                          hostMem, completion, refcon);
}

// reinit
//
//

IOReturn IOFWReadCommand::reinit(FWAddress devAddress,
	IOMemoryDescriptor *hostMem,
	FWDeviceCallback completion, void *refcon, bool failOnReset)
{
    return IOFWAsyncCommand::reinit(devAddress,
	hostMem, completion, refcon, failOnReset);
}

IOReturn IOFWReadCommand::reinit(UInt32 generation, FWAddress devAddress,
        IOMemoryDescriptor *hostMem,
        FWDeviceCallback completion, void *refcon)
{
    return IOFWAsyncCommand::reinit(generation, devAddress,
        hostMem, completion, refcon);
}

// execute
//
//

IOReturn IOFWReadCommand::execute()
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
			flags |= kIOFWWriteBlockRequest;
		}
	}
	
    fTrans = fControl->allocTrans(this);
    if(fTrans) {
        result = fControl->asyncRead(fGeneration, fNodeID, fAddressHi,
                        fAddressLo, fSpeed, fTrans->fTCode, transfer, this, (IOFWReadFlags)flags );
    }
    else {
    //    IOLog("IOFWReadCommand::execute: Out of tLabels?\n");
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
