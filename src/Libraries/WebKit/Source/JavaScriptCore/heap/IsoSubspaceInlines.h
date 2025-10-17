/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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

namespace JSC {

namespace GCClient {

ALWAYS_INLINE void* IsoSubspace::allocate(VM& vm, size_t cellSize, GCDeferralContext* deferralContext, AllocationFailureMode failureMode)
{
    return m_localAllocator.allocate(vm.heap, cellSize, deferralContext, failureMode);
}

} // namespace GCClient

inline void IsoSubspace::clearIsoCellSetBit(PreciseAllocation* preciseAllocation)
{
    unsigned lowerTierPreciseIndex = preciseAllocation->lowerTierPreciseIndex();
    m_cellSets.forEach(
        [&](IsoCellSet* set) {
            set->clearLowerTierPreciseCell(lowerTierPreciseIndex);
        });
}

inline void IsoSubspace::sweep()
{
    Subspace::sweepBlocks();
    // We sweep precise-allocations eagerly, but we do not free it immediately.
    // This part should be done by MarkedSpace::sweepPreciseAllocations.
    m_preciseAllocations.forEach([&](PreciseAllocation* allocation) {
        allocation->sweep();
    });
}

template<typename Func>
void IsoSubspace::forEachLowerTierPreciseFreeListedPreciseAllocation(const Func& func)
{
    m_lowerTierPreciseFreeList.forEach(func);
}

} // namespace JSC

