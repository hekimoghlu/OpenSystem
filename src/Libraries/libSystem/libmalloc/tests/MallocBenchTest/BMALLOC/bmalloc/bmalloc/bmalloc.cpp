/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#include "bmalloc.h"

#include "PerProcess.h"

namespace bmalloc { namespace api {

void* mallocOutOfLine(size_t size, HeapKind kind)
{
    return malloc(size, kind);
}

void freeOutOfLine(void* object, HeapKind kind)
{
    free(object, kind);
}

void* tryLargeZeroedMemalignVirtual(size_t alignment, size_t size, HeapKind kind)
{
    BASSERT(isPowerOfTwo(alignment));

    size_t pageSize = vmPageSize();
    alignment = roundUpToMultipleOf(pageSize, alignment);
    size = roundUpToMultipleOf(pageSize, size);

    kind = mapToActiveHeapKind(kind);
    Heap& heap = PerProcess<PerHeapKind<Heap>>::get()->at(kind);

    void* result;
    {
        std::unique_lock<Mutex> lock(Heap::mutex());
        result = heap.tryAllocateLarge(lock, alignment, size);
        if (result) {
            // Don't track this as dirty memory that dictates how we drive the scavenger.
            // FIXME: We should make it so that users of this API inform bmalloc which
            // pages they dirty:
            // https://bugs.webkit.org/show_bug.cgi?id=184207
            heap.externalDecommit(lock, result, size);
        }
    }

    if (result)
        vmZeroAndPurge(result, size);
    return result;
}

void freeLargeVirtual(void* object, size_t size, HeapKind kind)
{
    kind = mapToActiveHeapKind(kind);
    Heap& heap = PerProcess<PerHeapKind<Heap>>::get()->at(kind);
    std::unique_lock<Mutex> lock(Heap::mutex());
    // Balance out the externalDecommit when we allocated the zeroed virtual memory.
    heap.externalCommit(lock, object, size);
    heap.deallocateLarge(lock, object);
}

void scavenge()
{
    scavengeThisThread();

    PerProcess<Scavenger>::get()->scavenge();
}

bool isEnabled(HeapKind kind)
{
    kind = mapToActiveHeapKind(kind);
    std::unique_lock<Mutex> lock(Heap::mutex());
    return !PerProcess<PerHeapKind<Heap>>::getFastCase()->at(kind).debugHeap();
}

#if BOS(DARWIN)
void setScavengerThreadQOSClass(qos_class_t overrideClass)
{
    std::unique_lock<Mutex> lock(Heap::mutex());
    PerProcess<Scavenger>::get()->setScavengerThreadQOSClass(overrideClass);
}
#endif

void commitAlignedPhysical(void* object, size_t size, HeapKind kind)
{
    vmValidatePhysical(object, size);
    vmAllocatePhysicalPages(object, size);
    Heap& heap = PerProcess<PerHeapKind<Heap>>::get()->at(kind);
    heap.externalCommit(object, size);
}

void decommitAlignedPhysical(void* object, size_t size, HeapKind kind)
{
    vmValidatePhysical(object, size);
    vmDeallocatePhysicalPages(object, size);
    Heap& heap = PerProcess<PerHeapKind<Heap>>::get()->at(kind);
    heap.externalDecommit(object, size);
}

void enableMiniMode()
{
    PerProcess<Scavenger>::get()->enableMiniMode();
}

} } // namespace bmalloc::api

