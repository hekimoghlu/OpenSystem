/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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
#include "TinyBloomFilter.h"
#include <wtf/HashSet.h>

namespace JSC {

class MarkedBlock;

class MarkedBlockSet {
public:
    void add(MarkedBlock*);
    void remove(MarkedBlock*);

    TinyBloomFilter<uintptr_t> filter() const;
    const UncheckedKeyHashSet<MarkedBlock*>& set() const;

private:
    void recomputeFilter();

    TinyBloomFilter<uintptr_t> m_filter;
    UncheckedKeyHashSet<MarkedBlock*> m_set;
};

inline void MarkedBlockSet::add(MarkedBlock* block)
{
    m_filter.add(reinterpret_cast<uintptr_t>(block));
    m_set.add(block);
}

inline void MarkedBlockSet::remove(MarkedBlock* block)
{
    unsigned oldCapacity = m_set.capacity();
    m_set.remove(block);
    if (m_set.capacity() != oldCapacity) // Indicates we've removed a lot of blocks.
        recomputeFilter();
}

inline void MarkedBlockSet::recomputeFilter()
{
    TinyBloomFilter<uintptr_t> filter;
    for (auto* block : m_set)
        filter.add(reinterpret_cast<uintptr_t>(block));
    m_filter = filter;
}

inline TinyBloomFilter<uintptr_t> MarkedBlockSet::filter() const
{
    return m_filter;
}

inline const UncheckedKeyHashSet<MarkedBlock*>& MarkedBlockSet::set() const
{
    return m_set;
}

} // namespace JSC
