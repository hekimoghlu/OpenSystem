/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#include "HasSelectorFilter.h"

#include "ElementChildIteratorInlines.h"
#include "RuleFeature.h"
#include "SelectorFilter.h"
#include "StyleRule.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::Style {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HasSelectorFilter);

// FIXME: Support additional pseudo-classes.
static constexpr unsigned HoverSalt = 101;

HasSelectorFilter::HasSelectorFilter(const Element& element, Type type)
    : m_type(type)
{
    switch (type) {
    case Type::Descendants:
        for (auto& descendant : descendantsOfType<Element>(element))
            add(descendant);
        break;
    case Type::Children:
        for (auto& child : childrenOfType<Element>(element))
            add(child);
        break;
    }
}

auto HasSelectorFilter::typeForMatchElement(MatchElement matchElement) -> std::optional<Type>
{
    switch (matchElement) {
    case MatchElement::HasChild:
        return Type::Children;
    case MatchElement::HasDescendant:
        return Type::Descendants;
    default:
        return { };
    }
}

auto HasSelectorFilter::makeKey(const CSSSelector& hasSelector) -> Key
{
    SelectorFilter::CollectedSelectorHashes hashes;
    bool hasHoverInCompound = false;
    for (auto* simpleSelector = &hasSelector; simpleSelector; simpleSelector = simpleSelector->tagHistory()) {
        if (simpleSelector->match() == CSSSelector::Match::PseudoClass && simpleSelector->pseudoClass() == CSSSelector::PseudoClass::Hover)
            hasHoverInCompound = true;
        SelectorFilter::collectSimpleSelectorHash(hashes, *simpleSelector);
        if (!hashes.ids.isEmpty())
            break;
        if (simpleSelector->relation() != CSSSelector::Relation::Subselector)
            break;
    }

    auto pickKey = [&](auto& hashVector) -> Key {
        if (hashVector.isEmpty())
            return 0;
        if (hasHoverInCompound)
            return hashVector[0] * HoverSalt;
        return hashVector[0];
    };

    if (auto key = pickKey(hashes.ids))
        return key;
    if (auto key = pickKey(hashes.classes))
        return key;
    if (auto key = pickKey(hashes.attributes))
        return key;
    return pickKey(hashes.tags);
}

void HasSelectorFilter::add(const Element& element)
{
    Vector<unsigned, 4> elementHashes;
    SelectorFilter::collectElementIdentifierHashes(element, elementHashes);

    for (auto hash : elementHashes)
        m_filter.add(hash);

    if (element.hovered()) {
        for (auto hash : elementHashes)
            m_filter.add(hash * HoverSalt);
    }
}

}
