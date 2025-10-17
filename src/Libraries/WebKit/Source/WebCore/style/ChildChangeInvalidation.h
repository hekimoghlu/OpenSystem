/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#include "StyleInvalidator.h"
#include "StyleScope.h"
#include <wtf/HashSet.h>

namespace WebCore {
namespace Style {

class ChildChangeInvalidation {
public:
    ChildChangeInvalidation(ContainerNode&, const ContainerNode::ChildChange&);
    ~ChildChangeInvalidation();

    static void invalidateAfterFinishedParsingChildren(Element&);

private:
    void invalidateForHasBeforeMutation();
    void invalidateForHasAfterMutation();
    void invalidateAfterChange();
    void checkForSiblingStyleChanges();
    using MatchingHasSelectors = UncheckedKeyHashSet<const CSSSelector*>;
    enum class ChangedElementRelation : uint8_t { SelfOrDescendant, Sibling };
    void invalidateForChangedElement(Element&, MatchingHasSelectors&, ChangedElementRelation);
    void invalidateForChangeOutsideHasScope();

    template<typename Function> void traverseRemovedElements(Function&&);
    template<typename Function> void traverseAddedElements(Function&&);
    template<typename Function> void traverseRemainingExistingSiblings(Function&&);

    Element& parentElement() { return *m_parentElement; }

    Element* m_parentElement { nullptr };
    const ContainerNode::ChildChange& m_childChange;

    const bool m_isEnabled;
    const bool m_needsHasInvalidation;
    const bool m_wasEmpty;
};

inline ChildChangeInvalidation::ChildChangeInvalidation(ContainerNode& container, const ContainerNode::ChildChange& childChange)
    : m_parentElement(dynamicDowncast<Element>(container))
    , m_childChange(childChange)
    , m_isEnabled(m_parentElement && m_parentElement->needsStyleInvalidation())
    , m_needsHasInvalidation(m_isEnabled && Scope::forNode(*m_parentElement).usesHasPseudoClass())
    , m_wasEmpty(!container.firstChild())
{
    if (!m_isEnabled)
        return;

    if (m_needsHasInvalidation)
        invalidateForHasBeforeMutation();
}

inline ChildChangeInvalidation::~ChildChangeInvalidation()
{
    if (!m_isEnabled)
        return;

    if (m_needsHasInvalidation)
        invalidateForHasAfterMutation();

    invalidateAfterChange();
}

}
}
