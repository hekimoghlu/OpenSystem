/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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

#include "FormattingContext.h"
#include "LayoutElementBox.h"

namespace WebCore {
namespace Layout {

class LayoutContainingBlockChainIterator {
public:
    LayoutContainingBlockChainIterator() = default;
    LayoutContainingBlockChainIterator(const ElementBox*);
    const ElementBox& operator*() const { return *m_current; }
    const ElementBox* operator->() const { return m_current; }

    LayoutContainingBlockChainIterator& operator++();
    friend bool operator==(LayoutContainingBlockChainIterator, LayoutContainingBlockChainIterator) = default;

private:
    const ElementBox* m_current { nullptr };
};

class LayoutContainingBlockChainIteratorAdapter {
public:
    LayoutContainingBlockChainIteratorAdapter(const ElementBox&, const ElementBox* stayWithin = nullptr);
    auto begin() { return LayoutContainingBlockChainIterator(&m_containingBlock); }
    auto end() { return LayoutContainingBlockChainIterator(m_stayWithin); }

private:
    const ElementBox& m_containingBlock;
    const ElementBox* m_stayWithin { nullptr };
};

LayoutContainingBlockChainIteratorAdapter containingBlockChain(const Box&);
LayoutContainingBlockChainIteratorAdapter containingBlockChain(const Box&, const ElementBox& stayWithin);
LayoutContainingBlockChainIteratorAdapter containingBlockChainWithinFormattingContext(const Box&, const ElementBox& root);

inline LayoutContainingBlockChainIterator::LayoutContainingBlockChainIterator(const ElementBox* current)
    : m_current(current)
{
}

inline LayoutContainingBlockChainIterator& LayoutContainingBlockChainIterator::operator++()
{
    ASSERT(m_current);
    m_current = m_current->isInitialContainingBlock() ? nullptr : &FormattingContext::containingBlock(*m_current);
    return *this;
}

inline LayoutContainingBlockChainIteratorAdapter::LayoutContainingBlockChainIteratorAdapter(const ElementBox& containingBlock, const ElementBox* stayWithin)
    : m_containingBlock(containingBlock)
    , m_stayWithin(stayWithin)
{
}

inline LayoutContainingBlockChainIteratorAdapter containingBlockChain(const Box& layoutBox)
{
    return LayoutContainingBlockChainIteratorAdapter(FormattingContext::containingBlock(layoutBox));
}

inline LayoutContainingBlockChainIteratorAdapter containingBlockChain(const Box& layoutBox, const ElementBox& stayWithin)
{
    ASSERT(layoutBox.isDescendantOf(stayWithin));
    return LayoutContainingBlockChainIteratorAdapter(FormattingContext::containingBlock(layoutBox), &stayWithin);
}

inline LayoutContainingBlockChainIteratorAdapter containingBlockChainWithinFormattingContext(const Box& layoutBox, const ElementBox& root)
{
    ASSERT(root.establishesFormattingContext());
    return containingBlockChain(layoutBox, root);
}

}
}
