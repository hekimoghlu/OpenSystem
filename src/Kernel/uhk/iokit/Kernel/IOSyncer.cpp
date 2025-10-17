/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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
/* IOSyncer.cpp created by wgulland on 2000-02-02 */

#include <IOKit/IOLib.h>
#include <IOKit/IOSyncer.h>

OSDefineMetaClassAndStructors(IOSyncer, OSObject)

IOSyncer * IOSyncer::create(bool twoRetains)
{
	IOSyncer * me = new IOSyncer;

	if (me && !me->init(twoRetains)) {
		me->release();
		return NULL;
	}

	return me;
}

bool
IOSyncer::init(bool twoRetains)
{
	if (!OSObject::init()) {
		return false;
	}

	if (!(guardLock = IOSimpleLockAlloc())) {
		return false;
	}

	IOSimpleLockInit(guardLock);

	if (twoRetains) {
		retain();
	}

	fResult = kIOReturnSuccess;

	reinit();

	return true;
}

void
IOSyncer::reinit()
{
	IOInterruptState is = IOSimpleLockLockDisableInterrupt(guardLock);
	threadMustStop = true;
	IOSimpleLockUnlockEnableInterrupt(guardLock, is);
}

void
IOSyncer::free()
{
	// just in case a thread is blocked here:
	privateSignal();

	if (guardLock != NULL) {
		IOSimpleLockFree(guardLock);
	}

	OSObject::free();
}

IOReturn
IOSyncer::wait(bool autoRelease)
{
	IOInterruptState is = IOSimpleLockLockDisableInterrupt(guardLock);

	if (threadMustStop) {
		assert_wait((void *) &threadMustStop, false);
		IOSimpleLockUnlockEnableInterrupt(guardLock, is);
		thread_block(THREAD_CONTINUE_NULL);
	} else {
		IOSimpleLockUnlockEnableInterrupt(guardLock, is);
	}

	IOReturn result = fResult; // Pick up before auto deleting!

	if (autoRelease) {
		release();
	}

	return result;
}

void
IOSyncer::signal(IOReturn res, bool autoRelease)
{
	fResult = res;
	privateSignal();
	if (autoRelease) {
		release();
	}
}

void
IOSyncer::privateSignal()
{
	if (threadMustStop) {
		IOInterruptState is = IOSimpleLockLockDisableInterrupt(guardLock);
		threadMustStop = false;
		thread_wakeup_one((void *) &threadMustStop);
		IOSimpleLockUnlockEnableInterrupt(guardLock, is);
	}
}
