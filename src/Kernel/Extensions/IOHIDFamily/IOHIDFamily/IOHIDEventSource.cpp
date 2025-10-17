/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#include "IOHIDEventSource.h"

#undef  super
#define super IOEventSource

OSDefineMetaClassAndStructors(IOHIDEventSource, super);

IOHIDEventSource *
IOHIDEventSource::HIDEventSource(OSObject * inOwner, Action inAction)
{
    IOHIDEventSource * me = new IOHIDEventSource;
    
    if (me && !me->init(inOwner, inAction)) {
        me->release();
        return 0;
    }
    
    return me;
}


void
IOHIDEventSource::lock(void)
{
    closeGate();
}

void
IOHIDEventSource::unlock(void)
{
    openGate();
}

//
// When an event source is removed from a workloop the it results in
// the workLoop member variable of the event source being set to NULL.
// So if a thread tries to execute closeGate() after the removeEventSource()
// call there will be a kernel panic.
// If we try to work-around this problem by removing the event source in
// ::free() then it may lead to deadlocks as removeEventSource() grabs the
// workloop lock and free() can be called from any context.
// If we try to work-around this by addding an isInactive() check before the
// closeGate()/runAction() then there is a possibility that the thread
// can get preempted after the check and when closeGate() is called workLoop
// is set to NULL.
// So as a temprorary work-around, override setWorkLoop() such that if the
// workLoop is being set to NULL, skip setting it to NULL and set it to NULL
// only in IOHIDEventSource::free()
// This can be removed depending on the outcome of <rdar://problem/17666447> 

void
IOHIDEventSource::setWorkLoop(IOWorkLoop *inWorkLoop)
{
    if (!inWorkLoop) {
        disable();
    } else {
        super::setWorkLoop(inWorkLoop);
        workLoop->retain();
    }
}

void
IOHIDEventSource::free(void)
{
    OSSafeReleaseNULL(workLoop);
    
    super::free();
}
