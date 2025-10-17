/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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
#include "HTMLFormattingElementList.h"

#ifndef NDEBUG
#include <stdio.h>
#endif

namespace WebCore {

// Biblically, Noah's Ark only had room for two of each animal, but in the
// Book of Hixie (aka http://www.whatwg.org/specs/web-apps/current-work/multipage/parsing.html#list-of-active-formatting-elements),
// Noah's Ark of Formatting Elements can fit three of each element.
static const size_t kNoahsArkCapacity = 3;

HTMLFormattingElementList::HTMLFormattingElementList() = default;

HTMLFormattingElementList::~HTMLFormattingElementList() = default;

Element* HTMLFormattingElementList::closestElementInScopeWithName(ElementName targetElement)
{
    for (unsigned i = 1; i <= m_entries.size(); ++i) {
        const Entry& entry = m_entries[m_entries.size() - i];
        if (entry.isMarker())
            return nullptr;
        if (entry.stackItem().elementName() == targetElement)
            return &entry.element();
    }
    return nullptr;
}

bool HTMLFormattingElementList::contains(Element& element)
{
    return !!find(element);
}

auto HTMLFormattingElementList::find(Element& element) -> Entry*
{
    size_t index = m_entries.reverseFind(&element);
    if (index != notFound) {
        // This is somewhat of a hack, and is why this method can't be const.
        return &m_entries[index];
    }
    return nullptr;
}

auto HTMLFormattingElementList::bookmarkFor(Element& element) -> Bookmark
{
    size_t index = m_entries.reverseFind(&element);
    ASSERT(index != notFound);
    return Bookmark(at(index));
}

void HTMLFormattingElementList::swapTo(Ref<Element> oldElement, HTMLStackItem&& newItem, const Bookmark& bookmark)
{
    ASSERT(contains(oldElement));
    ASSERT(!contains(newItem.element()));
    if (!bookmark.hasBeenMoved()) {
        ASSERT(&bookmark.mark().element() == oldElement.ptr());
        bookmark.mark().replaceElement(WTFMove(newItem));
        return;
    }
    size_t index = &bookmark.mark() - &first();
    ASSERT_WITH_SECURITY_IMPLICATION(index <= size());
    m_entries.insert(index, WTFMove(newItem));
    remove(oldElement);
}

void HTMLFormattingElementList::append(HTMLStackItem&& item)
{
    ensureNoahsArkCondition(item);
    m_entries.append(WTFMove(item));
}

void HTMLFormattingElementList::remove(Element& element)
{
    size_t index = m_entries.reverseFind(&element);
    if (index != notFound)
        m_entries.remove(index);
}

void HTMLFormattingElementList::removeUpdatingBookmark(Element& element, Bookmark& bookmark)
{
    size_t index = m_entries.reverseFind(&element);
    if (index != notFound) {
        size_t bookmarkIndex = &bookmark.mark() - &first();
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(bookmarkIndex <= size());
        m_entries.remove(index);
        // Removing an element from the list can change the position of the bookmarked
        // item. Update the address pointed by the bookmark, when needed.
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
        if (bookmarkIndex > index)
            bookmark.m_mark--;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    }
}

void HTMLFormattingElementList::appendMarker()
{
    m_entries.append(Entry::MarkerEntry);
}

void HTMLFormattingElementList::clearToLastMarker()
{
    // http://www.whatwg.org/specs/web-apps/current-work/multipage/parsing.html#clear-the-list-of-active-formatting-elements-up-to-the-last-marker
    while (m_entries.size()) {
        bool shouldStop = m_entries.last().isMarker();
        m_entries.removeLast();
        if (shouldStop)
            break;
    }
}

static bool itemsHaveMatchingNames(const HTMLStackItem& a, const HTMLStackItem& b)
{
    if (a.elementName() != b.elementName())
        return false;

    if (a.elementName() != ElementName::Unknown)
        return true;

    return a.localName() == b.localName() && a.namespaceURI() == b.namespaceURI();
}

Vector<const HTMLStackItem*> HTMLFormattingElementList::tryToEnsureNoahsArkConditionQuickly(HTMLStackItem& newItem)
{
    if (m_entries.size() < kNoahsArkCapacity)
        return { };

    // Use a vector with inline capacity to avoid a malloc in the common case of quickly ensuring the condition.
    Vector<const HTMLStackItem*, 10> candidates;

    size_t newItemAttributeCount = newItem.attributes().size();

    for (size_t i = m_entries.size(); i; ) {
        --i;
        Entry& entry = m_entries[i];
        if (entry.isMarker())
            break;

        // Quickly reject obviously non-matching candidates.
        auto& candidate = entry.stackItem();
        if (!itemsHaveMatchingNames(newItem, candidate))
            continue;
        if (candidate.attributes().size() != newItemAttributeCount)
            continue;

        candidates.append(&candidate);
    }

    if (candidates.size() < kNoahsArkCapacity)
        return { }; // There's room for the new element in the ark. There's no need to copy out the remainingCandidates.

    return candidates.span();
}

void HTMLFormattingElementList::ensureNoahsArkCondition(HTMLStackItem& newItem)
{
    auto candidates = tryToEnsureNoahsArkConditionQuickly(newItem);
    if (candidates.isEmpty())
        return;

    // We pre-allocate and re-use this second vector to save one malloc per attribute that we verify.
    Vector<const HTMLStackItem*> remainingCandidates;
    remainingCandidates.reserveInitialCapacity(candidates.size());

    for (auto& attribute : newItem.attributes()) {
        for (auto* candidate : candidates) {
            // These properties should already have been checked by tryToEnsureNoahsArkConditionQuickly.
            ASSERT(newItem.attributes().size() == candidate->attributes().size());
            ASSERT(itemsHaveMatchingNames(newItem, *candidate));

            auto* candidateAttribute = candidate->findAttribute(attribute.name());
            if (candidateAttribute && candidateAttribute->value() == attribute.value())
                remainingCandidates.append(candidate);
        }

        if (remainingCandidates.size() < kNoahsArkCapacity)
            return;

        candidates.swap(remainingCandidates);
        remainingCandidates.shrink(0);
    }

    // Inductively, we shouldn't spin this loop very many times. It's possible,
    // however, that we will spin the loop more than once because of how the
    // formatting element list gets permuted.
    for (size_t i = kNoahsArkCapacity - 1; i < candidates.size(); ++i)
        remove(candidates[i]->protectedElement());
}

#if ENABLE(TREE_DEBUGGING)

void HTMLFormattingElementList::show()
{
    for (unsigned i = 1; i <= m_entries.size(); ++i) {
        const Entry& entry = m_entries[m_entries.size() - i];
        if (entry.isMarker())
            fprintf(stderr, "marker\n");
        else
            entry.element().showNode();
    }
}

#endif

}
