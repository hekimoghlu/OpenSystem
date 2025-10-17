/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#include "AccessibilitySVGRoot.h"

#include "AXObjectCache.h"
#include "ElementInlines.h"
#include "RenderObject.h"
#include "SVGDescElement.h"
#include "SVGElementTypeHelpers.h"
#include "SVGTitleElement.h"
#include "TypedElementDescendantIteratorInlines.h"

namespace WebCore {

AccessibilitySVGRoot::AccessibilitySVGRoot(AXID axID, RenderObject& renderer, AXObjectCache* cache)
    : AccessibilitySVGObject(axID, renderer, cache)
{
}

AccessibilitySVGRoot::~AccessibilitySVGRoot() = default;

Ref<AccessibilitySVGRoot> AccessibilitySVGRoot::create(AXID axID, RenderObject& renderer, AXObjectCache* cache)
{
    return adoptRef(*new AccessibilitySVGRoot(axID, renderer, cache));
}

void AccessibilitySVGRoot::setParent(AccessibilityRenderObject* parent)
{
    m_parent = parent;
}
    
AccessibilityObject* AccessibilitySVGRoot::parentObject() const
{
    // If a parent was set because this is a remote SVG resource, use that
    // but otherwise, we should rely on the standard render tree for the parent.
    if (m_parent)
        return m_parent.get();
    
    return AccessibilitySVGObject::parentObject();
}

AccessibilityRole AccessibilitySVGRoot::determineAccessibilityRole()
{
    if ((m_ariaRole = determineAriaRoleAttribute()) != AccessibilityRole::Unknown)
        return m_ariaRole;
    return AccessibilityRole::Generic;
}

bool AccessibilitySVGRoot::hasAccessibleContent() const
{
    auto* rootElement = this->element();
    if (!rootElement)
        return false;

    auto isAccessibleSVGElement = [] (const SVGElement& element) -> bool {
        // The presence of an SVGTitle or SVGDesc element is enough to deem the SVG hierarchy as accessible.
        if (is<SVGTitleElement>(element)
            || is<SVGDescElement>(element))
            return true;

        // Text content is accessible.
        if (element.isTextContent())
            return true;

        // If the role or aria-label attributes are specified, this is accessible.
        if (!element.attributeWithoutSynchronization(HTMLNames::roleAttr).isEmpty()
            || !element.attributeWithoutSynchronization(HTMLNames::aria_labelAttr).isEmpty())
            return true;

        return false;
    };

    auto* svgRootElement = dynamicDowncast<SVGElement>(*rootElement);
    if (svgRootElement && isAccessibleSVGElement(*svgRootElement))
        return true;

    // This SVG hierarchy is accessible if any of its descendants is accessible.
    for (const auto& descendant : descendantsOfType<SVGElement>(*rootElement)) {
        if (isAccessibleSVGElement(descendant))
            return true;
    }

    return false;
}

} // namespace WebCore
