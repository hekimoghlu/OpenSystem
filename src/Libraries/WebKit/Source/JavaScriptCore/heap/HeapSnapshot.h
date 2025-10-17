/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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

#include "HeapSnapshotBuilder.h"
#include "TinyBloomFilter.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {

class HeapSnapshot {
    WTF_MAKE_TZONE_ALLOCATED(HeapSnapshot);
public:
    HeapSnapshot(HeapSnapshot*);
    ~HeapSnapshot();

    HeapSnapshot* previous() const { return m_previous; }

    void appendNode(const HeapSnapshotNode&);
    void sweepCell(JSCell*);
    void shrinkToFit();
    void finalize();

    bool isEmpty() const { return m_nodes.isEmpty(); }
    std::optional<HeapSnapshotNode> nodeForCell(JSCell*);
    std::optional<HeapSnapshotNode> nodeForObjectIdentifier(unsigned objectIdentifier);

private:
    friend class HeapSnapshotBuilder;
    static constexpr intptr_t CellToSweepTag = 1;

    Vector<HeapSnapshotNode> m_nodes;
    TinyBloomFilter<uintptr_t> m_filter;
    HeapSnapshot* m_previous { nullptr };
    unsigned m_firstObjectIdentifier { 0 };
    unsigned m_lastObjectIdentifier { 0 };
    bool m_finalized { false };
    bool m_hasCellsToSweep { false };
};

} // namespace JSC
