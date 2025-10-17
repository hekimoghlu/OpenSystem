/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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

#include "Element.h"
#include <memory>
#include <wtf/BloomFilter.h>
#include <wtf/Vector.h>

namespace WebCore {

class CSSSelector;

class SelectorFilter {
public:
    void pushParent(Element* parent);
    void pushParentInitializingIfNeeded(Element& parent);
    void popParent();
    void popParentsUntil(Element* parent);
    bool parentStackIsEmpty() const { return m_parentStack.isEmpty(); }
    bool parentStackIsConsistent(const ContainerNode* parentNode) const;
    void parentStackReserveInitialCapacity(size_t initialCapacity) { m_parentStack.reserveInitialCapacity(initialCapacity); }

    using Hashes = std::array<unsigned, 4>;
    bool fastRejectSelector(const Hashes&) const;
    static Hashes collectHashes(const CSSSelector&);

    static void collectElementIdentifierHashes(const Element&, Vector<unsigned, 4>&);

    struct CollectedSelectorHashes {
        using HashVector = Vector<unsigned, 8>;
        HashVector ids;
        HashVector classes;
        HashVector tags;
        HashVector attributes;
    };
    static void collectSimpleSelectorHash(CollectedSelectorHashes&, const CSSSelector&);

    WEBCORE_EXPORT static CollectedSelectorHashes collectHashesForTesting(const CSSSelector&);

private:
    void initializeParentStack(Element& parent);
    enum class IncludeRightmost : bool { No, Yes };
    static void collectSelectorHashes(CollectedSelectorHashes&, const CSSSelector& rightmostSelector, IncludeRightmost);
    static Hashes chooseSelectorHashesForFilter(const CollectedSelectorHashes&);

    struct ParentStackFrame {
        ParentStackFrame() : element(0) { }
        ParentStackFrame(Element* element) : element(element) { }
        Element* element;
        Vector<unsigned, 4> identifierHashes;
    };
    Vector<ParentStackFrame> m_parentStack;

    // With 100 unique strings in the filter, 2^12 slot table has false positive rate of ~0.2%.
    static const unsigned bloomFilterKeyBits = 12;
    CountingBloomFilter<bloomFilterKeyBits> m_ancestorIdentifierFilter;
};

inline bool SelectorFilter::fastRejectSelector(const Hashes& hashes) const
{
    for (auto& hash : hashes) {
        if (!hash)
            return false;
        if (!m_ancestorIdentifierFilter.mayContain(hash))
            return true;
    }
    return false;
}

} // namespace WebCore
