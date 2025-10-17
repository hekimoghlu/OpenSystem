/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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
#pragma once

#include "Subspace.h"

namespace JSC {

class CompleteSubspace final : public Subspace {
public:
    JS_EXPORT_PRIVATE CompleteSubspace(CString name, Heap&, const HeapCellType&, AlignedMemoryAllocator*);
    JS_EXPORT_PRIVATE ~CompleteSubspace() final;

    // In some code paths, we need it to be a compile error to call the virtual version of one of
    // these functions. That's why we do final methods the old school way.
    
    // FIXME: Currently subspaces speak of BlockDirectories as "allocators", but that's temporary.
    // https://bugs.webkit.org/show_bug.cgi?id=181559
    Allocator allocatorFor(size_t, AllocatorForMode);
    Allocator allocatorForNonInline(size_t, AllocatorForMode);

    void* allocate(VM&, size_t, GCDeferralContext*, AllocationFailureMode);
    void* reallocatePreciseAllocationNonVirtual(VM&, HeapCell*, size_t, GCDeferralContext*, AllocationFailureMode);
    
    static constexpr ptrdiff_t offsetOfAllocatorForSizeStep() { return OBJECT_OFFSETOF(CompleteSubspace, m_allocatorForSizeStep); }
    
    Allocator* allocatorForSizeStep() { return &m_allocatorForSizeStep[0]; }

private:
    JS_EXPORT_PRIVATE Allocator allocatorForSlow(size_t);
    
    // These slow paths are concerned with large allocations and allocator creation.
    JS_EXPORT_PRIVATE void* allocateSlow(VM&, size_t, GCDeferralContext*, AllocationFailureMode);
    void* tryAllocateSlow(VM&, size_t, GCDeferralContext*);
    
    std::array<Allocator, MarkedSpace::numSizeClasses> m_allocatorForSizeStep;
    Vector<std::unique_ptr<BlockDirectory>> m_directories;
    Vector<std::unique_ptr<LocalAllocator>> m_localAllocators;
};

ALWAYS_INLINE Allocator CompleteSubspace::allocatorFor(size_t size, AllocatorForMode mode)
{
    if (size <= MarkedSpace::largeCutoff) {
        Allocator result = m_allocatorForSizeStep[MarkedSpace::sizeClassToIndex(size)];
        switch (mode) {
        case AllocatorForMode::MustAlreadyHaveAllocator:
            RELEASE_ASSERT(result);
            break;
        case AllocatorForMode::EnsureAllocator:
            if (!result)
                return allocatorForSlow(size);
            break;
        case AllocatorForMode::AllocatorIfExists:
            break;
        }
        return result;
    }
    RELEASE_ASSERT(mode != AllocatorForMode::MustAlreadyHaveAllocator);
    return Allocator();
}

} // namespace JSC

