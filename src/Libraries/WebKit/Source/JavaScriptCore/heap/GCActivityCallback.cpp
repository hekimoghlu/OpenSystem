/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#include "config.h"
#include "GCActivityCallback.h"

#include "HeapInlines.h"
#include "VM.h"

namespace JSC {

bool GCActivityCallback::s_shouldCreateGCTimer = true;

const double timerSlop = 2.0; // Fudge factor to avoid performance cost of resetting timer.

GCActivityCallback::GCActivityCallback(JSC::Heap& heap, Synchronousness synchronousness)
    : GCActivityCallback(heap.vm(), synchronousness)
{
}

GCActivityCallback::GCActivityCallback(VM& vm, Synchronousness synchronousness)
    : Base(vm)
    , m_synchronousness(synchronousness)
{
}

GCActivityCallback::~GCActivityCallback() = default;

void GCActivityCallback::doWork(VM& vm)
{
    if (!isEnabled())
        return;
    
    ASSERT(vm.currentThreadIsHoldingAPILock());
    JSC::Heap& heap = vm.heap;
    if (heap.isDeferred()) {
        scheduleTimer(0_s);
        return;
    }

    doCollection(vm);
}

void GCActivityCallback::scheduleTimer(Seconds newDelay)
{
    if (newDelay * timerSlop > m_delay)
        return;
    Seconds delta = m_delay - newDelay;
    m_delay = newDelay;
    if (auto timeUntilFire = this->timeUntilFire())
        newDelay = *timeUntilFire - delta;
    setTimeUntilFire(newDelay);
}

void GCActivityCallback::didAllocate(JSC::Heap& heap, size_t bytes)
{
    // The first byte allocated in an allocation cycle will report 0 bytes to didAllocate. 
    // We pretend it's one byte so that we don't ignore this allocation entirely.
    if (!bytes)
        bytes = 1;
    double bytesExpectedToReclaim = static_cast<double>(bytes) * deathRate(heap);
    Seconds newDelay = lastGCLength(heap) / gcTimeSlice(bytesExpectedToReclaim);
    scheduleTimer(newDelay);
}

void GCActivityCallback::willCollect()
{
    cancel();
}

void GCActivityCallback::cancel()
{
    m_delay = s_decade;
    cancelTimer();
}

}

