/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#if ENABLE(DFG_JIT)

#include "DFGAbstractHeap.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/PrintStream.h>

namespace JSC { namespace DFG {

class Graph;
struct Node;

// FIXME: If we ever want to compare if two nodes clobber each other, we should
// have a SmallClobberSet, which just keeps an array of the AbstractHeaps and
// satisfies overlaps() requests by looping over all of them. This will probably
// be faster than a full UncheckedKeyHashMap in a lot of cases. Or, maybe, we could have
// ClobberSet be smart and use a vector so long as it was small.

class ClobberSet {
public:
    ClobberSet();
    ~ClobberSet();
    
    bool isEmpty() const { return m_clobbers.isEmpty(); }
    
    void add(AbstractHeap);
    void addAll(const ClobberSet&);
    bool overlaps(AbstractHeap) const;
    void clear();
    
    // Calls useful for debugging the ClobberSet.
    // Do not call for non debugging purpose. Otherwise, you must handle DOMState hierarchy carefully.
    
    UncheckedKeyHashSet<AbstractHeap> direct() const;
    UncheckedKeyHashSet<AbstractHeap> super() const;
    
    void dump(PrintStream&) const;
    
private:
    bool contains(AbstractHeap) const;

    UncheckedKeyHashSet<AbstractHeap> setOf(bool direct) const;
    
    // Maps heap to:
    // true --> it's a direct clobber
    // false --> it's just a supertype of a direct clobber
    UncheckedKeyHashMap<AbstractHeap, bool> m_clobbers;
};

class ClobberSetAdd {
public:
    ClobberSetAdd(ClobberSet& set)
        : m_set(set)
    {
    }
    
    void operator()(AbstractHeap heap) const
    {
        m_set.add(heap);
    }
private:
    ClobberSet& m_set;
};

class ClobberSetOverlaps {
public:
    ClobberSetOverlaps(const ClobberSet& set)
        : m_set(set)
        , m_result(false)
    {
    }
    
    void operator()(AbstractHeap heap) const
    {
        m_result |= m_set.overlaps(heap);
    }
    
    bool result() const { return m_result; }
    
private:
    const ClobberSet& m_set;
    mutable bool m_result;
};

void addReads(Graph&, Node*, ClobberSet&);
void addWrites(Graph&, Node*, ClobberSet&);
void addReadsAndWrites(Graph&, Node*, ClobberSet& reads, ClobberSet& writes);

ClobberSet writeSet(Graph&, Node*);

bool readsOverlap(Graph&, Node*, ClobberSet&);
bool writesOverlap(Graph&, Node*, ClobberSet&);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
