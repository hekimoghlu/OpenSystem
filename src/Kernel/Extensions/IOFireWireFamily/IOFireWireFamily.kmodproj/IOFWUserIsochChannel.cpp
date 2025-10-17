/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
 *  IOFWUserIsochChannel.cpp
 *  IOFireWireFamily
 *
 *  Created by noggin on Tue May 15 2001.
 *  Copyright (c) 2001 Apple Computer, Inc. All rights reserved.
 *
 */

#import <IOKit/firewire/IOFireWireController.h>
#import <IOKit/firewire/IOFWCommand.h>
#import <IOKit/firewire/IOFWLocalIsochPort.h>
#import <IOKit/firewire/IOFWDCLProgram.h>

#import "IOFireWireUserClient.h"
#import "IOFWUserIsochChannel.h"

OSDefineMetaClassAndStructors(IOFWUserIsochChannel, IOFWIsochChannel)

bool IOFWUserIsochChannel::init(	
	IOFireWireController *		control, 
	bool 						doIRM,
	UInt32 						packetSize, 
	IOFWSpeed 					prefSpeed )
{
	DebugLog("IOFWUserIsochChannel<%p>::init - packetSize = %d, doIRM = %d\n", this, packetSize, doIRM );
	
	return super::init( control, doIRM, packetSize, prefSpeed, &IOFWUserIsochChannel::isochChannel_ForceStopHandler, this ) ;
}

IOReturn
IOFWUserIsochChannel::allocateChannel()
{
	// maybe we should call user space lib here?
//	IOLog("IOFWUserIsochChannel::allocateChannel called!\n") ;
	return kIOReturnUnsupported ;
}

IOReturn
IOFWUserIsochChannel::releaseChannel()
{
	// maybe we should call user space lib here?
//	IOLog("IOFWUserIsochChannel::releaseChannel called!\n") ;
	return kIOReturnUnsupported ;
}


IOReturn
IOFWUserIsochChannel::start()
{
	// maybe we should call user space lib here?
//	IOLog("IOFWUserIsochChannel::start called!\n") ;
	return kIOReturnUnsupported ;
}

IOReturn
IOFWUserIsochChannel::stop()
{
	// maybe we should call user space lib here?
//	IOLog("IOFWUserIsochChannel::stop called!\n") ;
	return kIOReturnUnsupported ;
}

IOReturn
IOFWUserIsochChannel::allocateListenerPorts()
{
	IOFWIsochPort*		listen;
	IOReturn			result 			= kIOReturnSuccess ;
	OSIterator*			listenIterator	= OSCollectionIterator::withCollection(fListeners) ;

	if(listenIterator) {
		listenIterator->reset();
		while( (listen = (IOFWIsochPort *) listenIterator->getNextObject()) && (result == kIOReturnSuccess)) {
			result = listen->allocatePort(fSpeed, fChannel);
		}
		listenIterator->release();
	}
	
	return result ;
}

IOReturn
IOFWUserIsochChannel::allocateTalkerPort()
{
	IOReturn	result	= kIOReturnSuccess ;

	if(fTalker)
		result = fTalker->allocatePort(fSpeed, fChannel);
	
	return result ;
}

void
IOFWUserIsochChannel::s_exporterCleanup ( IOFWUserIsochChannel * channel )
{
	DebugLog( "IOFWUserIsochChannel::s_exporterCleanup - channel = %p\n", channel) ;
	
	channel->fControl->removeAllocatedChannel( channel ) ;

	channel->stop() ;
	channel->releaseChannel() ;
}

IOReturn
IOFWUserIsochChannel::isochChannel_ForceStopHandler( void * self, IOFWIsochChannel*, UInt32 stopCondition )
{
	IOFWUserIsochChannel * me = (IOFWUserIsochChannel*)self;
	
#if INFO
	natural_t userProc = me->fStopRefCon ? ((natural_t*)me->fStopRefCon)[ kIOAsyncCalloutFuncIndex ] : 0 ;
	natural_t userRef =  me->fStopRefCon ? ((natural_t*)me->fStopRefCon)[ kIOAsyncCalloutRefconIndex ] : 0 ;

	InfoLog("+ IOFireWireUserClient::s_IsochChannel_ForceStopHandler() -- fStopRefCon=%p, userProc=%p, userRef=0x%x\n", me->fStopRefCon, userProc, userRef ) ;
#endif


	if ( !me->getUserAsyncRef() )
	{
		return kIOReturnSuccess ;
	}
	
	return IOFireWireUserClient::sendAsyncResult64( (io_user_reference_t *)me->getUserAsyncRef(), stopCondition, NULL, 0 ) ;
}
