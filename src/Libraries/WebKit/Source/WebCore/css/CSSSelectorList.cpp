/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
#include "config.h"
#include "CSSSelectorList.h"

#include "CommonAtomStrings.h"
#include "MutableCSSSelector.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CSSSelectorList);

CSSSelectorList::CSSSelectorList(const CSSSelectorList& other)
{
    unsigned otherComponentCount = other.componentCount();
    if (!otherComponentCount)
        return;

    m_selectorArray = makeUniqueArray<CSSSelector>(otherComponentCount);
    for (unsigned i = 0; i < otherComponentCount; ++i)
        new (NotNull, &m_selectorArray[i]) CSSSelector(other.m_selectorArray[i]);
}

CSSSelectorList::CSSSelectorList(MutableCSSSelectorList&& selectorVector)
{
    ASSERT_WITH_SECURITY_IMPLICATION(!selectorVector.isEmpty());

    size_t flattenedSize = 0;
    for (size_t i = 0; i < selectorVector.size(); ++i) {
        for (auto* selector = selectorVector[i].get(); selector; selector = selector->tagHistory())
            ++flattenedSize;
    }
    ASSERT(flattenedSize);
    m_selectorArray = makeUniqueArray<CSSSelector>(flattenedSize);
    size_t arrayIndex = 0;
    for (size_t i = 0; i < selectorVector.size(); ++i) {
        auto* first = selectorVector[i].get();
        auto* current = first;
        while (current) {
            {
                // Move item from the parser selector vector into m_selectorArray without invoking destructor (Ugh.)
                CSSSelector* currentSelector = current->releaseSelector().release();
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
                memcpy(static_cast<void*>(&m_selectorArray[arrayIndex]), static_cast<void*>(currentSelector), sizeof(CSSSelector));
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

                // Free the underlying memory without invoking the destructor.
                operator delete (currentSelector);
            }
            if (current != first)
                m_selectorArray[arrayIndex].setNotFirstInTagHistory();
            current = current->tagHistory();
            ASSERT(!m_selectorArray[arrayIndex].isLastInSelectorList() || (flattenedSize == arrayIndex + 1));
            if (current)
                m_selectorArray[arrayIndex].setNotLastInTagHistory();
            ++arrayIndex;
        }
        ASSERT(m_selectorArray[arrayIndex - 1].isLastInTagHistory());
    }
    ASSERT(flattenedSize == arrayIndex);
    m_selectorArray[arrayIndex - 1].setLastInSelectorList();
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
unsigned CSSSelectorList::componentCount() const
{
    if (!m_selectorArray)
        return 0;
    CSSSelector* current = m_selectorArray.get();
    while (!current->isLastInSelectorList())
        ++current;
    return (current - m_selectorArray.get()) + 1;
}

unsigned CSSSelectorList::listSize() const
{
    if (!m_selectorArray)
        return 0;
    unsigned size = 1;
    CSSSelector* current = m_selectorArray.get();
    while (!current->isLastInSelectorList()) {
        if (current->isLastInTagHistory())
            ++size;
        ++current;
    }
    return size;
}
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

String CSSSelectorList::selectorsText() const
{
    StringBuilder result;
    buildSelectorsText(result);
    return result.toString();
}

void CSSSelectorList::buildSelectorsText(StringBuilder& stringBuilder) const
{
    stringBuilder.append(interleave(*this, [](auto& subSelector) { return subSelector.selectorText(); }, ", "_s));
}

template <typename Functor>
static bool forEachTagSelector(Functor& functor, const CSSSelector* selector)
{
    ASSERT(selector);

    do {
        if (functor(selector))
            return true;
        if (const CSSSelectorList* selectorList = selector->selectorList()) {
            for (const auto& subSelector : *selectorList) {
                if (forEachTagSelector(functor, &subSelector))
                    return true;
            }
        }
    } while ((selector = selector->tagHistory()));

    return false;
}

template <typename Functor>
static bool forEachSelector(Functor& functor, const CSSSelectorList& selectorList)
{
    for (const auto& selector : selectorList) {
        if (forEachTagSelector(functor, &selector))
            return true;
    }

    return false;
}

bool CSSSelectorList::hasExplicitNestingParent() const
{
    auto functor = [](auto* selector) {
        return selector->hasExplicitNestingParent();
    };

    return forEachSelector(functor, *this);
}

bool CSSSelectorList::hasOnlyNestingSelector() const
{
    if (componentCount() != 1)
        return false;

    auto singleSelector = first();

    if (!singleSelector)
        return false;
    
    // Selector should be a single selector
    if (singleSelector->tagHistory())
        return false;

    return singleSelector->match() == CSSSelector::Match::NestingParent;
}

} // namespace WebCore
