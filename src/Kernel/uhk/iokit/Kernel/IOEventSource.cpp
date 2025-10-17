/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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
 *  Copyright (c) 1998 Apple Computer, Inc.  All rights reserved.
 *
 *  HISTORY
 *   1998-7-13	Godfrey van der Linden(gvdl)
 *       Created.
 *  ]*/

#define IOKIT_ENABLE_SHARED_PTR

#include <IOKit/IOLib.h>

#include <IOKit/IOEventSource.h>
#include <IOKit/IOWorkLoop.h>
#include <libkern/Block.h>

#define super OSObject

OSDefineMetaClassAndAbstractStructors(IOEventSource, OSObject)

OSMetaClassDefineReservedUnused(IOEventSource, 0);
OSMetaClassDefineReservedUnused(IOEventSource, 1);
OSMetaClassDefineReservedUnused(IOEventSource, 2);
OSMetaClassDefineReservedUnused(IOEventSource, 3);
OSMetaClassDefineReservedUnused(IOEventSource, 4);
OSMetaClassDefineReservedUnused(IOEventSource, 5);
OSMetaClassDefineReservedUnused(IOEventSource, 6);
OSMetaClassDefineReservedUnused(IOEventSource, 7);

bool
IOEventSource::checkForWork()
{
	return false;
}

/* inline function implementations */

#if IOKITSTATS

#define IOStatisticsRegisterCounter() \
do { \
	reserved->counter = IOStatistics::registerEventSource(inOwner); \
} while (0)

#define IOStatisticsUnregisterCounter() \
do { \
	if (reserved) \
	        IOStatistics::unregisterEventSource(reserved->counter); \
} while (0)

#define IOStatisticsOpenGate() \
do { \
	IOStatistics::countOpenGate(reserved->counter); \
} while (0)

#define IOStatisticsCloseGate() \
do { \
	IOStatistics::countCloseGate(reserved->counter); \
} while (0)

#else

#define IOStatisticsRegisterCounter()
#define IOStatisticsUnregisterCounter()
#define IOStatisticsOpenGate()
#define IOStatisticsCloseGate()

#endif /* IOKITSTATS */

void
IOEventSource::signalWorkAvailable()
{
	workLoop->signalWorkAvailable();
}

void
IOEventSource::openGate()
{
	IOStatisticsOpenGate();
	workLoop->openGate();
}

void
IOEventSource::closeGate()
{
	workLoop->closeGate();
	IOStatisticsCloseGate();
}

bool
IOEventSource::tryCloseGate()
{
	bool res;
	if ((res = workLoop->tryCloseGate())) {
		IOStatisticsCloseGate();
	}
	return res;
}

int
IOEventSource::sleepGate(void *event, UInt32 type)
{
	int res;
	IOStatisticsOpenGate();
	res = workLoop->sleepGate(event, type);
	IOStatisticsCloseGate();
	return res;
}

int
IOEventSource::sleepGate(void *event, AbsoluteTime deadline, UInt32 type)
{
	int res;
	IOStatisticsOpenGate();
	res = workLoop->sleepGate(event, deadline, type);
	IOStatisticsCloseGate();
	return res;
}

void
IOEventSource::wakeupGate(void *event, bool oneThread)
{
	workLoop->wakeupGate(event, oneThread);
}


bool
IOEventSource::init(OSObject *inOwner,
    Action inAction)
{
	if (!inOwner) {
		return false;
	}

	owner = inOwner;

	if (!super::init()) {
		return false;
	}

	(void) setAction(inAction);
	enabled = true;

	if (!reserved) {
		reserved = IOMallocType(ExpansionData);
	}

	IOStatisticsRegisterCounter();

	return true;
}

void
IOEventSource::free( void )
{
	IOStatisticsUnregisterCounter();

	if ((kActionBlock & flags) && actionBlock) {
		Block_release(actionBlock);
	}

	if (reserved) {
		IOFreeType(reserved, ExpansionData);
	}

	super::free();
}

void
IOEventSource::setRefcon(void *newrefcon)
{
	refcon = newrefcon;
}

void *
IOEventSource::getRefcon() const
{
	return refcon;
}

IOEventSource::Action
IOEventSource::getAction() const
{
	if (kActionBlock & flags) {
		return NULL;
	}
	return action;
}

IOEventSource::ActionBlock
IOEventSource::getActionBlock(ActionBlock) const
{
	if (kActionBlock & flags) {
		return actionBlock;
	}
	return NULL;
}

void
IOEventSource::setAction(Action inAction)
{
	if ((kActionBlock & flags) && actionBlock) {
		Block_release(actionBlock);
	}
	action = inAction;
	flags &= ~kActionBlock;
}

void
IOEventSource::setActionBlock(ActionBlock block)
{
	if ((kActionBlock & flags) && actionBlock) {
		Block_release(actionBlock);
	}
	actionBlock = Block_copy(block);
	flags |= kActionBlock;
}

IOEventSource *
IOEventSource::getNext() const
{
	return eventChainNext;
};

void
IOEventSource::setNext(IOEventSource *inNext)
{
	eventChainNext = inNext;
}

void
IOEventSource::enable()
{
	enabled = true;
	if (workLoop) {
		return signalWorkAvailable();
	}
}

void
IOEventSource::disable()
{
	enabled = false;
}

bool
IOEventSource::isEnabled() const
{
	return enabled;
}

void
IOEventSource::setWorkLoop(IOWorkLoop *inWorkLoop)
{
	if (!inWorkLoop) {
		disable();
	}
	workLoop = inWorkLoop;
}

IOWorkLoop *
IOEventSource::getWorkLoop() const
{
	return workLoop;
}

bool
IOEventSource::onThread() const
{
	return (workLoop != NULL) && workLoop->onThread();
}
