/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#include "EdenGCActivityCallback.h"

#include "VM.h"

namespace JSC {

EdenGCActivityCallback::EdenGCActivityCallback(JSC::Heap& heap, Synchronousness synchronousness)
    : GCActivityCallback(heap, synchronousness)
{
}

EdenGCActivityCallback::~EdenGCActivityCallback() = default;

void EdenGCActivityCallback::doCollection(VM& vm)
{
    setDidGCRecently(false);
    vm.heap.collect(m_synchronousness, CollectionScope::Eden);
}

Seconds EdenGCActivityCallback::lastGCLength(JSC::Heap& heap)
{
    return heap.lastEdenGCLength();
}

double EdenGCActivityCallback::deathRate(JSC::Heap& heap)
{
    size_t sizeBefore = heap.sizeBeforeLastEdenCollection();
    size_t sizeAfter = heap.sizeAfterLastEdenCollection();
    if (!sizeBefore)
        return 1.0;
    if (sizeAfter > sizeBefore) {
        // GC caused the heap to grow(!)
        // This could happen if the we visited more extra memory than was reported allocated.
        // We don't return a negative death rate, since that would schedule the next GC in the past.
        return 0;
    }
    return static_cast<double>(sizeBefore - sizeAfter) / static_cast<double>(sizeBefore);
}

double EdenGCActivityCallback::gcTimeSlice(size_t bytes)
{
    return std::min((static_cast<double>(bytes) / MB) * Options::percentCPUPerMBForEdenTimer(), Options::collectionTimerMaxPercentCPU());
}

} // namespace JSC
