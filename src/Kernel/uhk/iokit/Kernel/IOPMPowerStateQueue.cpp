/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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
#include "IOPMPowerStateQueue.h"

#define super IOEventSource
OSDefineMetaClassAndStructors( IOPMPowerStateQueue, IOEventSource )

IOPMPowerStateQueue * IOPMPowerStateQueue::PMPowerStateQueue(
	OSObject * inOwner, Action inAction )
{
	IOPMPowerStateQueue * me = new IOPMPowerStateQueue;

	if (me && !me->init(inOwner, inAction)) {
		me->release();
		return NULL;
	}

	return me;
}

bool
IOPMPowerStateQueue::init( OSObject * inOwner, Action inAction )
{
	if (!inAction || !(super::init(inOwner, inAction))) {
		return false;
	}

	queue_init( &queueHead );

	queueLock = IOLockAlloc();
	if (!queueLock) {
		return false;
	}

	return true;
}

bool
IOPMPowerStateQueue::submitPowerEvent(
	uint32_t eventType,
	void *   arg0,
	uint64_t arg1 )
{
	PowerEventEntry * entry;

	entry = IOMallocType(PowerEventEntry);

	entry->eventType = eventType;
	entry->arg0 = arg0;
	entry->arg1 = arg1;

	IOLockLock(queueLock);
	queue_enter(&queueHead, entry, PowerEventEntry *, chain);
	IOLockUnlock(queueLock);
	signalWorkAvailable();

	return true;
}

bool
IOPMPowerStateQueue::checkForWork( void )
{
	IOPMPowerStateQueueAction queueAction = (IOPMPowerStateQueueAction) action;
	PowerEventEntry * entry;

	IOLockLock(queueLock);
	while (!queue_empty(&queueHead)) {
		queue_remove_first(&queueHead, entry, PowerEventEntry *, chain);
		IOLockUnlock(queueLock);

		(*queueAction)(owner, entry->eventType, entry->arg0, entry->arg1);
		IOFreeType(entry, PowerEventEntry);

		IOLockLock(queueLock);
	}
	IOLockUnlock(queueLock);

	return false;
}
