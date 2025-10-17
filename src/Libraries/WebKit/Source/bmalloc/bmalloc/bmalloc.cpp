/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

#include "DebugHeap.h"
#include "Environment.h"
#include "PerProcess.h"
#include "ProcessCheck.h"

#if BENABLE(LIBPAS)
#include "bmalloc_heap_config.h"
#include "pas_page_sharing_pool.h"
#include "pas_probabilistic_guard_malloc_allocator.h"
#include "pas_scavenger.h"
#include "pas_thread_local_cache.h"
#endif

namespace bmalloc { namespace api {

#if BUSE(LIBPAS)
namespace {
static const bmalloc_type primitiveGigacageType = BMALLOC_TYPE_INITIALIZER(1, 1, "Primitive Gigacage");
} // anonymous namespace

pas_primitive_heap_ref gigacageHeaps[static_cast<size_t>(Gigacage::NumberOfKinds)] = {
    BMALLOC_AUXILIARY_HEAP_REF_INITIALIZER(&primitiveGigacageType),
};
#endif

void* mallocOutOfLine(size_t size, CompactAllocationMode mode, HeapKind kind)
{
    return malloc(size, mode, kind);
}

void freeOutOfLine(void* object, HeapKind kind)
{
    free(object, kind);
}

void* tryLargeZeroedMemalignVirtual(size_t requiredAlignment, size_t requestedSize, CompactAllocationMode mode, HeapKind kind)
{
    RELEASE_BASSERT(isPowerOfTwo(requiredAlignment));

    size_t pageSize = vmPageSize();
    size_t alignment = roundUpToMultipleOf(pageSize, requiredAlignment);
    size_t size = roundUpToMultipleOf(pageSize, requestedSize);
    RELEASE_BASSERT(alignment >= requiredAlignment);
    RELEASE_BASSERT(size >= requestedSize);

    void* result;
    if (auto* debugHeap = DebugHeap::tryGet())
        result = debugHeap->memalignLarge(alignment, size);
    else {
#if BUSE(LIBPAS)
        result = tryMemalign(alignment, size, mode, kind);
#else
        BUNUSED(mode);
        kind = mapToActiveHeapKind(kind);
        Heap& heap = PerProcess<PerHeapKind<Heap>>::get()->at(kind);

        UniqueLockHolder lock(Heap::mutex());
        result = heap.allocateLarge(lock, alignment, size, FailureAction::ReturnNull);
        if (result) {
            // Don't track this as dirty memory that dictates how we drive the scavenger.
            // FIXME: We should make it so that users of this API inform bmalloc which
            // pages they dirty:
            // https://bugs.webkit.org/show_bug.cgi?id=184207
            heap.externalDecommit(lock, result, size);
        }
#endif
    }

    if (result)
        vmZeroAndPurge(result, size);
    return result;
}

void freeLargeVirtual(void* object, size_t size, HeapKind kind)
{
#if BUSE(LIBPAS)
    BUNUSED(size);
    BUNUSED(kind);
    if (auto* debugHeap = DebugHeap::tryGet()) {
        debugHeap->freeLarge(object);
        return;
    }
    bmalloc_deallocate_inline(object);
#else
    if (auto* debugHeap = DebugHeap::tryGet()) {
        debugHeap->freeLarge(object);
        return;
    }
    kind = mapToActiveHeapKind(kind);
    Heap& heap = PerProcess<PerHeapKind<Heap>>::get()->at(kind);
    UniqueLockHolder lock(Heap::mutex());
    // Balance out the externalDecommit when we allocated the zeroed virtual memory.
    heap.externalCommit(lock, object, size);
    heap.deallocateLarge(lock, object);
#endif
}

void scavengeThisThread()
{
#if BENABLE(LIBPAS)
    pas_thread_local_cache_shrink(pas_thread_local_cache_try_get(),
                                  pas_lock_is_not_held);
#endif
#if !BUSE(LIBPAS)
    if (!DebugHeap::tryGet()) {
        for (unsigned i = numHeaps; i--;)
            Cache::scavenge(static_cast<HeapKind>(i));
        IsoTLS::scavenge();
    }
#endif
}

void scavenge()
{
#if BENABLE(LIBPAS)
    pas_scavenger_run_synchronously_now();
#endif
    scavengeThisThread();
    if (DebugHeap* debugHeap = DebugHeap::tryGet())
        debugHeap->scavenge();
    else {
#if !BUSE(LIBPAS)
        Scavenger::get()->scavenge();
#endif
    }
}

bool isEnabled(HeapKind)
{
    return !Environment::get()->isDebugHeapEnabled();
}

#if BOS(DARWIN)
void setScavengerThreadQOSClass(qos_class_t overrideClass)
{
#if BENABLE(LIBPAS)
    pas_scavenger_set_requested_qos_class(overrideClass);
#endif
#if !BUSE(LIBPAS)
    if (!DebugHeap::tryGet()) {
        UniqueLockHolder lock(Heap::mutex());
        Scavenger::get()->setScavengerThreadQOSClass(overrideClass);
    }
#endif
}
#endif

void commitAlignedPhysical(void* object, size_t size, HeapKind kind)
{
    vmValidatePhysical(object, size);
    vmAllocatePhysicalPages(object, size);
#if BUSE(LIBPAS)
    BUNUSED(kind);
#else
    if (!DebugHeap::tryGet())
        PerProcess<PerHeapKind<Heap>>::get()->at(kind).externalCommit(object, size);
#endif
}

void decommitAlignedPhysical(void* object, size_t size, HeapKind kind)
{
    vmValidatePhysical(object, size);
    vmDeallocatePhysicalPages(object, size);
#if BUSE(LIBPAS)
    BUNUSED(kind);
#else
    if (!DebugHeap::tryGet())
        PerProcess<PerHeapKind<Heap>>::get()->at(kind).externalDecommit(object, size);
#endif
}

void enableMiniMode()
{
#if BENABLE(LIBPAS)
    if (!shouldAllowMiniMode())
        return;

    // Speed up the scavenger.
    pas_scavenger_period_in_milliseconds = 5.;
    pas_scavenger_max_epoch_delta = 5ll * 1000ll * 1000ll;

    // Do eager scavenging anytime pages are allocated or committed.
    pas_physical_page_sharing_pool_balancing_enabled = true;
    pas_physical_page_sharing_pool_balancing_enabled_for_utility = true;

    // Switch to bitfit allocation for anything that isn't isoheaped.
    bmalloc_intrinsic_runtime_config.base.max_segregated_object_size = 0;
    bmalloc_intrinsic_runtime_config.base.max_bitfit_object_size = UINT_MAX;
    bmalloc_primitive_runtime_config.base.max_segregated_object_size = 0;
    bmalloc_primitive_runtime_config.base.max_bitfit_object_size = UINT_MAX;
#endif
#if !BUSE(LIBPAS)
    if (!DebugHeap::tryGet())
        Scavenger::get()->enableMiniMode();
#endif
}

void disableScavenger()
{
#if BENABLE(LIBPAS)
    pas_scavenger_suspend();
#endif
#if !BUSE(LIBPAS)
    if (!DebugHeap::tryGet())
        Scavenger::get()->disable();
#endif
}

void forceEnablePGM(uint16_t guardMallocRate)
{
#if BUSE(LIBPAS)
    pas_probabilistic_guard_malloc_initialize_pgm_as_enabled(guardMallocRate);
#else
    BUNUSED_PARAM(guardMallocRate);
#endif
}

} } // namespace bmalloc::api

