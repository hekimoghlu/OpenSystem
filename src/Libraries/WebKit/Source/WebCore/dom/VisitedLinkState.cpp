/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
#include "VisitedLinkState.h"

#include "ElementIterator.h"
#include "FrameDestructionObserverInlines.h"
#include "HTMLAnchorElementInlines.h"
#include "LocalFrame.h"
#include "Page.h"
#include "SVGAElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGNames.h"
#include "TypedElementDescendantIteratorInlines.h"
#include "VisitedLinkStore.h"
#include "XLinkNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(VisitedLinkState);

using namespace HTMLNames;

inline static const AtomString linkAttribute(const Element& element)
{
    if (!element.isLink())
        return nullAtom();
    if (element.isHTMLElement())
        return element.attributeWithoutSynchronization(HTMLNames::hrefAttr);
    if (element.isSVGElement())
        return element.getAttribute(SVGNames::hrefAttr, XLinkNames::hrefAttr);
    return nullAtom();
}

VisitedLinkState::VisitedLinkState(Document& document)
    : m_document(document)
{
}

void VisitedLinkState::invalidateStyleForAllLinks()
{
    if (m_linksCheckedForVisitedState.isEmpty())
        return;
    Ref document = m_document.get();
    for (Ref element : descendantsOfType<Element>(document.get())) {
        if (element->isLink())
            element->invalidateStyleForSubtree();
    }
}

inline static std::optional<SharedStringHash> linkHashForElement(const Element& element)
{
    if (auto anchor = dynamicDowncast<HTMLAnchorElement>(element))
        return anchor->visitedLinkHash();
    if (auto anchor = dynamicDowncast<SVGAElement>(element))
        return anchor->visitedLinkHash();
    return std::nullopt;
}

void VisitedLinkState::invalidateStyleForLink(SharedStringHash linkHash)
{
    if (!m_linksCheckedForVisitedState.contains(linkHash))
        return;
    Ref document = m_document.get();
    for (Ref element : descendantsOfType<Element>(document.get())) {
        if (element->isLink() && linkHashForElement(element) == linkHash)
            element->invalidateStyleForSubtree();
    }
}

InsideLink VisitedLinkState::determineLinkStateSlowCase(const Element& element)
{
    ASSERT(element.isLink());

    auto attribute = linkAttribute(element);
    if (attribute.isNull())
        return InsideLink::NotInside;

    auto hashIfFound = linkHashForElement(element);

    if (!hashIfFound)
        return attribute.isEmpty() ? InsideLink::InsideVisited : InsideLink::InsideUnvisited;

    auto hash = *hashIfFound;

    // An empty href (hash==0) refers to the document itself which is always visited. It is useful to check this explicitly so
    // that visited links can be tested in platform independent manner, without explicit support in the test harness.
    if (!hash)
        return InsideLink::InsideVisited;

    RefPtr frame = element.document().frame();
    if (!frame)
        return InsideLink::InsideUnvisited;

    RefPtr page = frame->page();
    if (!page)
        return InsideLink::InsideUnvisited;

    m_linksCheckedForVisitedState.add(hash);

    if (!page->visitedLinkStore().isLinkVisited(*page, hash, element.document().baseURL(), attribute))
        return InsideLink::InsideUnvisited;

    return InsideLink::InsideVisited;
}

} // namespace WebCore
