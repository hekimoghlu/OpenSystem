/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
 *	Copyright (c) 2000 Apple Computer, Inc.  All rights reserved.
 *
 *	HISTORY
 *
 *	2001-01-17	gvdl	Re-implement on IOCommandGate::commandSleep
 *	10/9/2000	CJS	Created IOCommandPool class and implementation
 *
 */

#define IOKIT_ENABLE_SHARED_PTR

#include <IOKit/IOCommandPool.h>
#include <libkern/c++/OSSharedPtr.h>

#define super OSObject
OSDefineMetaClassAndStructors(IOCommandPool, OSObject);
OSMetaClassDefineReservedUnused(IOCommandPool, 0);
OSMetaClassDefineReservedUnused(IOCommandPool, 1);
OSMetaClassDefineReservedUnused(IOCommandPool, 2);
OSMetaClassDefineReservedUnused(IOCommandPool, 3);
OSMetaClassDefineReservedUnused(IOCommandPool, 4);
OSMetaClassDefineReservedUnused(IOCommandPool, 5);
OSMetaClassDefineReservedUnused(IOCommandPool, 6);
OSMetaClassDefineReservedUnused(IOCommandPool, 7);

//--------------------------------------------------------------------------
//	withWorkLoop -	primary initializer and factory method
//--------------------------------------------------------------------------

OSSharedPtr<IOCommandPool>
IOCommandPool::
withWorkLoop(IOWorkLoop *inWorkLoop)
{
	OSSharedPtr<IOCommandPool> me = OSMakeShared<IOCommandPool>();

	if (me && !me->initWithWorkLoop(inWorkLoop)) {
		return nullptr;
	}

	return me;
}


bool
IOCommandPool::
initWithWorkLoop(IOWorkLoop *inWorkLoop)
{
	assert(inWorkLoop);

	if (!super::init()) {
		return false;
	}

	queue_init(&fQueueHead);

	fSerializer = IOCommandGate::commandGate(this);
	assert(fSerializer);
	if (!fSerializer) {
		return false;
	}

	if (kIOReturnSuccess != inWorkLoop->addEventSource(fSerializer.get())) {
		return false;
	}

	return true;
}

//--------------------------------------------------------------------------
//	commandPool & init -	obsolete initializer and factory method
//--------------------------------------------------------------------------

OSSharedPtr<IOCommandPool>
IOCommandPool::
commandPool(IOService * inOwner, IOWorkLoop *inWorkLoop, UInt32 inSize)
{
	OSSharedPtr<IOCommandPool> me = OSMakeShared<IOCommandPool>();

	if (me && !me->init(inOwner, inWorkLoop, inSize)) {
		return nullptr;
	}

	return me;
}

bool
IOCommandPool::
init(IOService */* inOwner */, IOWorkLoop *inWorkLoop, UInt32 /* inSize */)
{
	return initWithWorkLoop(inWorkLoop);
}


//--------------------------------------------------------------------------
//	free -	free all allocated resources
//--------------------------------------------------------------------------

void
IOCommandPool::free(void)
{
	if (fSerializer) {
		// remove our event source from owner's workloop
		IOWorkLoop *wl = fSerializer->getWorkLoop();
		if (wl) {
			wl->removeEventSource(fSerializer.get());
		}

		fSerializer.reset();
	}

	// Tell our superclass to cleanup too
	super::free();
}


//--------------------------------------------------------------------------
//	getCommand -	Gets a command from the pool. Pass true in
//			blockForCommand if you want your thread to sleep
//			waiting for resources
//--------------------------------------------------------------------------

OSSharedPtr<IOCommand>
IOCommandPool::getCommand(bool blockForCommand)
{
	IOReturn     result  = kIOReturnSuccess;
	IOCommand *command = NULL;

	IOCommandGate::Action func = OSMemberFunctionCast(
		IOCommandGate::Action, this, &IOCommandPool::gatedGetCommand);
	result = fSerializer->
	    runAction(func, (void *) &command, (void *) blockForCommand);
	if (kIOReturnSuccess == result) {
		return OSSharedPtr<IOCommand>(command, OSNoRetain);
	} else {
		return NULL;
	}
}


//--------------------------------------------------------------------------
//	gatedGetCommand -	Static callthrough function
//				(on safe side of command gate)
//--------------------------------------------------------------------------

IOReturn
IOCommandPool::
gatedGetCommand(IOCommand **command, bool blockForCommand)
{
	while (queue_empty(&fQueueHead)) {
		if (!blockForCommand) {
			return kIOReturnNoResources;
		}

		fSleepers++;
		fSerializer->commandSleep(&fSleepers, THREAD_UNINT);
	}

	queue_remove_first(&fQueueHead,
	    *command, IOCommand *, fCommandChain);
	return kIOReturnSuccess;
}


//--------------------------------------------------------------------------
//	returnCommand -		Returns command to the pool.
//--------------------------------------------------------------------------

void
IOCommandPool::
returnCommand(IOCommand *command)
{
	IOCommandGate::Action func = OSMemberFunctionCast(
		IOCommandGate::Action, this, &IOCommandPool::gatedReturnCommand);
	(void) fSerializer->runAction(func, (void *) command);
}


//--------------------------------------------------------------------------
//	gatedReturnCommand -	Callthrough function
//                              (on safe side of command gate)
//--------------------------------------------------------------------------

IOReturn
IOCommandPool::
gatedReturnCommand(IOCommand *command)
{
	queue_enter_first(&fQueueHead, command, IOCommand *, fCommandChain);
	if (fSleepers) {
		fSerializer->commandWakeup(&fSleepers, /* oneThread */ true);
		fSleepers--;
	}
	return kIOReturnSuccess;
}
