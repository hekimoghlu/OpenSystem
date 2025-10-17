/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
 *	IOFireWireAVCLocalNode.h
 *
 * Implementation of class to initialize the Local node's AVC Target mode support
 *
 */

#include <IOKit/avc/IOFireWireAVCLocalNode.h>

OSDefineMetaClassAndStructors(IOFireWireAVCLocalNode, IOService);

#pragma mark -
#pragma mark ¥¥¥ IOService methods ¥¥¥

bool IOFireWireAVCLocalNode::start(IOService *provider)
{
	//IOLog( "IOFireWireAVCLocalNode::start\n");

    fDevice = OSDynamicCast(IOFireWireNub, provider);
	if(!fDevice)
        return false;
	
    if (!IOService::start(provider))
        return false;

    fPCRSpace = IOFireWirePCRSpace::getPCRAddressSpace(fDevice->getBus());
    if(!fPCRSpace)
        return false;
    fPCRSpace->activate();

    fAVCTargetSpace = IOFireWireAVCTargetSpace::getAVCTargetSpace(fDevice->getController());
    if(!fAVCTargetSpace)
        return false;
    fAVCTargetSpace->activateWithUserClient((IOFireWireAVCProtocolUserClient*)0xFFFFFFFF);

	// Enable the communication between the PCR space and the Target space objects
	fPCRSpace->setAVCTargetSpacePointer(fAVCTargetSpace);
	
    registerService();

	fStarted = true;
	
    return true;
}

bool IOFireWireAVCLocalNode::finalize(IOOptionBits options)
{
	//IOLog( "IOFireWireAVCLocalNode::finalize\n");

	return IOService::finalize(options);
}

void IOFireWireAVCLocalNode::stop(IOService *provider)
{
	//IOLog( "IOFireWireAVCLocalNode::stop\n");

	IOService::stop(provider);
}

void IOFireWireAVCLocalNode::free(void)
{
	//IOLog( "IOFireWireAVCLocalNode::free\n");

    if(fPCRSpace)
	{
        fPCRSpace->deactivate();
        fPCRSpace->release();
    }

    if(fAVCTargetSpace)
	{
        fAVCTargetSpace->deactivateWithUserClient((IOFireWireAVCProtocolUserClient*)0xFFFFFFFF);
        fAVCTargetSpace->release();
    }
	
	return IOService::free();
}

IOReturn IOFireWireAVCLocalNode::message(UInt32 type, IOService *provider, void *argument)
{
    IOReturn res = kIOReturnUnsupported;

	//IOLog( "IOFireWireAVCLocalNode::message\n");

	switch (type)
	{
		case kIOMessageServiceIsTerminated:
		case kIOMessageServiceIsRequestingClose:
		case kIOMessageServiceIsResumed:
			res = kIOReturnSuccess;
			break;

		// This message is received when a bus-reset start happens!
		case kIOMessageServiceIsSuspended:
			res = kIOReturnSuccess;
			if((fStarted == true) && (fPCRSpace))
				fPCRSpace->clearAllP2PConnections();
			break;

		default:
			break;
	}
	
	messageClients(type);

    return res;
}


