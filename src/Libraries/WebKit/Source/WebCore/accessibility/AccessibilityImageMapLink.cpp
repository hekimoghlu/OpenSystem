/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
#include "AccessibilityImageMapLink.h"

#include "AXObjectCache.h"
#include "AccessibilityRenderObject.h"
#include "Document.h"
#include "HTMLNames.h"
#include "RenderBoxModelObject.h"

namespace WebCore {
    
using namespace HTMLNames;

AccessibilityImageMapLink::AccessibilityImageMapLink(AXID axID)
    : AccessibilityMockObject(axID)
    , m_areaElement(nullptr)
    , m_mapElement(nullptr)
{
}

AccessibilityImageMapLink::~AccessibilityImageMapLink() = default;

Ref<AccessibilityImageMapLink> AccessibilityImageMapLink::create(AXID axID)
{
    return adoptRef(*new AccessibilityImageMapLink(axID));
}

void AccessibilityImageMapLink::setHTMLAreaElement(HTMLAreaElement* element)
{
    if (element == m_areaElement)
        return;
    m_areaElement = element;
    // AccessibilityImageMapLink::determineAccessibilityRole() depends on m_areaElement, so re-compute it now.
    updateRole();
}

AccessibilityObject* AccessibilityImageMapLink::parentObject() const
{
    if (m_parent)
        return m_parent.get();
    
    if (!m_mapElement.get() || !m_mapElement->renderer())
        return nullptr;
    
    return m_mapElement->document().axObjectCache()->getOrCreate(m_mapElement->renderer());
}
    
AccessibilityRole AccessibilityImageMapLink::determineAccessibilityRole()
{
    if (!m_areaElement)
        return AccessibilityRole::WebCoreLink;
    
    const AtomString& ariaRole = getAttribute(roleAttr);
    if (!ariaRole.isEmpty())
        return AccessibilityObject::ariaRoleToWebCoreRole(ariaRole);

    return AccessibilityRole::WebCoreLink;
}
    
Element* AccessibilityImageMapLink::actionElement() const
{
    return anchorElement();
}
    
Element* AccessibilityImageMapLink::anchorElement() const
{
    return m_areaElement.get();
}

URL AccessibilityImageMapLink::url() const
{
    if (!m_areaElement.get())
        return URL();
    
    return m_areaElement->href();
}

void AccessibilityImageMapLink::accessibilityText(Vector<AccessibilityText>& textOrder) const
{
    String description = this->description();
    if (!description.isEmpty())
        textOrder.append(AccessibilityText(description, AccessibilityTextSource::Alternative));

    const AtomString& titleText = getAttribute(titleAttr);
    if (!titleText.isEmpty())
        textOrder.append(AccessibilityText(titleText, AccessibilityTextSource::TitleTag));

    const AtomString& summary = getAttribute(summaryAttr);
    if (!summary.isEmpty())
        textOrder.append(AccessibilityText(summary, AccessibilityTextSource::Summary));
}

String AccessibilityImageMapLink::description() const
{
    auto ariaLabel = getAttributeTrimmed(aria_labelAttr);
    if (!ariaLabel.isEmpty())
        return ariaLabel;

    const auto& alt = getAttribute(altAttr);
    if (!alt.isEmpty())
        return alt;

    return { };
}

String AccessibilityImageMapLink::title() const
{
    const AtomString& title = getAttribute(titleAttr);
    if (!title.isEmpty())
        return title;
    const AtomString& summary = getAttribute(summaryAttr);
    if (!summary.isEmpty())
        return summary;

    return String();
}

RenderElement* AccessibilityImageMapLink::imageMapLinkRenderer() const
{
    if (!m_mapElement || !m_areaElement)
        return nullptr;

    if (auto* parent = dynamicDowncast<AccessibilityRenderObject>(m_parent.get()))
        return downcast<RenderElement>(parent->renderer());
    return m_mapElement->renderer();
}

void AccessibilityImageMapLink::detachFromParent()
{
    AccessibilityMockObject::detachFromParent();
    m_areaElement = nullptr;
    m_mapElement = nullptr;
}

Path AccessibilityImageMapLink::elementPath() const
{
    auto renderer = imageMapLinkRenderer();
    if (!renderer)
        return Path();
    
    return m_areaElement->computePath(renderer);
}
    
LayoutRect AccessibilityImageMapLink::elementRect() const
{
    auto renderer = imageMapLinkRenderer();
    if (!renderer)
        return LayoutRect();
    
    return m_areaElement->computeRect(renderer);
}
    
} // namespace WebCore
