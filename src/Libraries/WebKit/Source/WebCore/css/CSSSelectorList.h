/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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

#include "CSSSelector.h"
#include <iterator>
#include <memory>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueArray.h>

namespace WebCore {

class MutableCSSSelector;
using MutableCSSSelectorList = Vector<std::unique_ptr<MutableCSSSelector>>;

class CSSSelectorList {
    WTF_MAKE_TZONE_ALLOCATED(CSSSelectorList);
public:
    CSSSelectorList() = default;
    CSSSelectorList(const CSSSelectorList&);
    CSSSelectorList(CSSSelectorList&&) = default;
    explicit CSSSelectorList(MutableCSSSelectorList&&);
    explicit CSSSelectorList(UniqueArray<CSSSelector>&& array)
        : m_selectorArray(WTFMove(array)) { }

    bool isEmpty() const { return !m_selectorArray; }
    const CSSSelector* first() const { return m_selectorArray.get(); }
    static const CSSSelector* next(const CSSSelector*);
    const CSSSelector* selectorAt(size_t index) const { return &m_selectorArray[index]; }

    size_t indexOfNextSelectorAfter(size_t index) const
    {
        const CSSSelector* current = selectorAt(index);
        current = next(current);
        if (!current)
            return notFound;
        return current - m_selectorArray.get();
    }

    struct const_iterator {
        friend class CSSSelectorList;
        using value_type = CSSSelector;
        using difference_type = std::ptrdiff_t;
        using pointer = const CSSSelector*;
        using reference = const CSSSelector&;
        using iterator_category = std::forward_iterator_tag;
        reference operator*() const { return *m_ptr; }
        pointer operator->() const { return m_ptr; }
        bool operator==(const const_iterator&) const = default;
        const_iterator() = default;
        const_iterator(pointer ptr) : m_ptr(ptr) { };
        const_iterator& operator++()
        {
            m_ptr = CSSSelectorList::next(m_ptr);
            return *this;
        }
    private:
        pointer m_ptr = nullptr;
    };
    const_iterator begin() const { return { first() }; };
    const_iterator end() const { return { }; }

    bool hasExplicitNestingParent() const;
    bool hasOnlyNestingSelector() const;

    String selectorsText() const;
    void buildSelectorsText(StringBuilder&) const;

    unsigned componentCount() const;
    unsigned listSize() const;

    CSSSelectorList& operator=(CSSSelectorList&&) = default;

private:
    // End of a multipart selector is indicated by m_isLastInTagHistory bit in the last item.
    // End of the array is indicated by m_isLastInSelectorList bit in the last item.
    UniqueArray<CSSSelector> m_selectorArray;
};

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
inline const CSSSelector* CSSSelectorList::next(const CSSSelector* current)
{
    // Skip subparts of compound selectors.
    while (!current->isLastInTagHistory())
        current++;
    return current->isLastInSelectorList() ? 0 : current + 1;
}
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

} // namespace WebCore
