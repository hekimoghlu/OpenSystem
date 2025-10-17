/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

#include "MarkedBlock.h"
#include <wtf/BitSet.h>
#include <wtf/ConcurrentVector.h>
#include <wtf/FastBitVector.h>
#include <wtf/Noncopyable.h>
#include <wtf/Nonmovable.h>
#include <wtf/SentinelLinkedList.h>
#include <wtf/SharedTask.h>

namespace JSC {

class HeapCell;
class IsoSubspace;

// Create a set of cells that are in an IsoSubspace. This allows concurrent O(1) set insertion and
// removal. Each such set should be thought of as a 0.8% increase in object size for objects in that
// IsoSubspace (it's like adding 1 bit every 16 bytes, or 1 bit every 128 bits).
class IsoCellSet final : public BasicRawSentinelNode<IsoCellSet> {
    WTF_MAKE_NONCOPYABLE(IsoCellSet);
    WTF_MAKE_NONMOVABLE(IsoCellSet);
public:
    IsoCellSet(IsoSubspace& subspace);
    ~IsoCellSet();
    
    bool add(HeapCell* cell); // Returns true if the cell was newly added.
    
    bool remove(HeapCell* cell); // Returns true if the cell was previously present and got removed.
    
    bool contains(HeapCell* cell) const;
    
    JS_EXPORT_PRIVATE Ref<SharedTask<MarkedBlock::Handle*()>> parallelNotEmptyMarkedBlockSource();
    
    // This will have to do a combined search over whatever Subspace::forEachMarkedCell uses and
    // our m_blocksWithBits.
    template<typename Func>
    void forEachMarkedCell(const Func&);

    template<typename Visitor, typename Func>
    Ref<SharedTask<void(Visitor&)>> forEachMarkedCellInParallel(const Func&);
    
    template<typename Func>
    void forEachLiveCell(const Func&);
    
private:
    friend class IsoSubspace;
    
    WTF::BitSet<MarkedBlock::atomsPerBlock>* addSlow(unsigned blockIndex);
    
    void didResizeBits(unsigned newSize);
    void didRemoveBlock(unsigned blockIndex);
    void sweepToFreeList(MarkedBlock::Handle*);
    void clearLowerTierPreciseCell(unsigned);
    
    WTF::BitSet<MarkedBlock::maxNumberOfLowerTierPreciseCells> m_lowerTierPreciseBits;

    IsoSubspace& m_subspace;
    
    // Idea: sweeping to free-list clears bits for those cells that were free-listed. The first time
    // we add a cell in a block, that block gets a free-list. Unless we do something that obviously
    // clears all bits for a block, we keep it set in blocksWithBits.
    
    FastBitVector m_blocksWithBits;
    ConcurrentVector<std::unique_ptr<WTF::BitSet<MarkedBlock::atomsPerBlock>>> m_bits;
};

} // namespace JSC

