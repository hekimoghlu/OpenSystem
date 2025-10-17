/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
#include "SVGMPathElement.h"

#include "Document.h"
#include "SVGAnimateMotionElement.h"
#include "SVGDocumentExtensions.h"
#include "SVGElementTypeHelpers.h"
#include "SVGNames.h"
#include "SVGPathElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGMPathElement);

inline SVGMPathElement::SVGMPathElement(const QualifiedName& tagName, Document& document)
    : SVGElement(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGURIReference(this)
{
    ASSERT(hasTagName(SVGNames::mpathTag));
}

Ref<SVGMPathElement> SVGMPathElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGMPathElement(tagName, document));
}

SVGMPathElement::~SVGMPathElement()
{
    clearResourceReferences();
}

void SVGMPathElement::buildPendingResource()
{
    clearResourceReferences();
    if (!isConnected())
        return;

    auto target = SVGURIReference::targetElementFromIRIString(href(), treeScopeForSVGReferences());
    if (!target.element) {
        // Do not register as pending if we are already pending this resource.
        auto& treeScope = treeScopeForSVGReferences();
        if (treeScope.isPendingSVGResource(*this, target.identifier))
            return;

        if (!target.identifier.isEmpty()) {
            treeScope.addPendingSVGResource(target.identifier, *this);
            ASSERT(hasPendingResources());
        }
    } else if (RefPtr svgElement = dynamicDowncast<SVGElement>(*target.element))
        svgElement->addReferencingElement(*this);

    targetPathChanged();
}

void SVGMPathElement::clearResourceReferences()
{
    removeElementReference();
}

Node::InsertedIntoAncestorResult SVGMPathElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    auto result = SVGElement::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (insertionType.connectedToDocument)
        return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
    return result;
}

void SVGMPathElement::didFinishInsertingNode()
{
    SVGElement::didFinishInsertingNode();
    buildPendingResource();
}

void SVGMPathElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    SVGElement::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (removalType.disconnectedFromDocument)
        clearResourceReferences();
}

void SVGMPathElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    SVGURIReference::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGMPathElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (SVGURIReference::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        buildPendingResource();
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
}

RefPtr<SVGPathElement> SVGMPathElement::pathElement()
{
    auto target = targetElementFromIRIString(href(), treeScopeForSVGReferences());
    return dynamicDowncast<SVGPathElement>(target.element);
}

void SVGMPathElement::targetPathChanged()
{
    if (RefPtr animateMotionElement = dynamicDowncast<SVGAnimateMotionElement>(parentNode()))
        animateMotionElement->updateAnimationPath();
}

} // namespace WebCore
