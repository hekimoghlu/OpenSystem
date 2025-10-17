/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
/* IOFWSyncer.cpp created by wgulland on 2000-02-02 */

#include <IOKit/IOLib.h>

#include <IOKit/firewire/IOFWSyncer.h>

OSDefineMetaClassAndStructors(IOFWSyncer, OSObject)

IOFWSyncer * IOFWSyncer::create(bool twoRetains)
{
    IOFWSyncer * me = OSTypeAlloc( IOFWSyncer );

    if (me && !me->init(twoRetains)) {
        me->release();
        return 0;
    }

    return me;
}

bool IOFWSyncer::init(bool twoRetains)
{
    if (!OSObject::init())
        return false;

    if (!(guardLock = IOSimpleLockAlloc()) )
        return false;
	
    IOSimpleLockInit(guardLock);

    if(twoRetains)
	retain();

    fResult = kIOReturnSuccess;

    reinit();

    return true;
}

void IOFWSyncer::reinit()
{
    IOInterruptState is = IOSimpleLockLockDisableInterrupt(guardLock);
    threadMustStop = true;
    IOSimpleLockUnlockEnableInterrupt(guardLock, is);
}

void IOFWSyncer::free()
{
    // just in case a thread is blocked here:
    privateSignal();

    if (guardLock != NULL)
       IOSimpleLockFree(guardLock);

    OSObject::free();
}

IOReturn IOFWSyncer::wait(bool autoRelease)
{
    IOInterruptState is = IOSimpleLockLockDisableInterrupt(guardLock);

    if (threadMustStop) {
	assert_wait((void *) &threadMustStop, false);
    	IOSimpleLockUnlockEnableInterrupt(guardLock, is);
        thread_block(THREAD_CONTINUE_NULL);
    }
    else
        IOSimpleLockUnlockEnableInterrupt(guardLock, is);

    IOReturn result = fResult;	// Pick up before auto deleting!

    if(autoRelease)
	release();

    return result;
}

void IOFWSyncer::signal(IOReturn res, bool autoRelease)
{
    fResult = res;
    privateSignal();
    if(autoRelease)
	release();
}

void IOFWSyncer::privateSignal()
{
    if (threadMustStop) {
         IOInterruptState is = IOSimpleLockLockDisableInterrupt(guardLock);
         threadMustStop = false;
         thread_wakeup_one((void *) &threadMustStop);
         IOSimpleLockUnlockEnableInterrupt(guardLock, is);
    }
}
