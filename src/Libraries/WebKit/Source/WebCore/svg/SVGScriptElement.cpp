/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include "SVGScriptElement.h"

#include "Document.h"
#include "Event.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGScriptElement);

inline SVGScriptElement::SVGScriptElement(const QualifiedName& tagName, Document& document, bool wasInsertedByParser, bool alreadyStarted)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGURIReference(this)
    , ScriptElement(*this, wasInsertedByParser, alreadyStarted)
    , m_loadEventTimer(*this, &SVGElement::loadEventTimerFired)
{
    ASSERT(hasTagName(SVGNames::scriptTag));
}

Ref<SVGScriptElement> SVGScriptElement::create(const QualifiedName& tagName, Document& document, bool insertedByParser)
{
    return adoptRef(*new SVGScriptElement(tagName, document, insertedByParser, false));
}

void SVGScriptElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGURIReference::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGScriptElement::svgAttributeChanged(const QualifiedName& attrName)
{
    InstanceInvalidationGuard guard(*this);

    if (SVGURIReference::isKnownAttribute(attrName)) {
        handleSourceAttribute(href());
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

Node::InsertedIntoAncestorResult SVGScriptElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    auto result1 = SVGElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    auto result2 = ScriptElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    return result1 == InsertedIntoAncestorResult::NeedsPostInsertionCallback ? result1 : result2;
}

void SVGScriptElement::didFinishInsertingNode()
{
    ScriptElement::didFinishInsertingNode();
}

void SVGScriptElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);
    ScriptElement::childrenChanged(change);
}

void SVGScriptElement::finishParsingChildren()
{
    SVGElement::finishParsingChildren();
    ScriptElement::finishParsingChildren();
}

void SVGScriptElement::addSubresourceAttributeURLs(ListHashSet<URL>& urls) const
{
    SVGElement::addSubresourceAttributeURLs(urls);

    addSubresourceURL(urls, document().completeURL(href()));
}
Ref<Element> SVGScriptElement::cloneElementWithoutAttributesAndChildren(TreeScope& treeScope)
{
    return adoptRef(*new SVGScriptElement(tagQName(), treeScope.documentScope(), false, alreadyStarted()));
}

void SVGScriptElement::dispatchErrorEvent()
{
    setErrorOccurred(true);
    ScriptElement::dispatchErrorEvent();
}

}
