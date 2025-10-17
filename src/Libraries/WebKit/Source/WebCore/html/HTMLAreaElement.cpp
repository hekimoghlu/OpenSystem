/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#include "HTMLAreaElement.h"

#include "AffineTransform.h"
#include "ElementInlines.h"
#include "HTMLImageElement.h"
#include "HTMLMapElement.h"
#include "HTMLParserIdioms.h"
#include "HitTestResult.h"
#include "LocalFrame.h"
#include "NodeName.h"
#include "Path.h"
#include "RenderImage.h"
#include "RenderView.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLAreaElement);

using namespace HTMLNames;

inline HTMLAreaElement::HTMLAreaElement(const QualifiedName& tagName, Document& document)
    : HTMLAnchorElement(tagName, document)
{
    ASSERT(hasTagName(areaTag));
}

HTMLAreaElement::~HTMLAreaElement() = default;

Ref<HTMLAreaElement> HTMLAreaElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new HTMLAreaElement(tagName, document));
}

void HTMLAreaElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::shapeAttr:
        if (equalLettersIgnoringASCIICase(newValue, "default"_s))
            m_shape = Shape::Default;
        else if (equalLettersIgnoringASCIICase(newValue, "circle"_s) || equalLettersIgnoringASCIICase(newValue, "circ"_s))
            m_shape = Shape::Circle;
        else if (equalLettersIgnoringASCIICase(newValue, "poly"_s) || equalLettersIgnoringASCIICase(newValue, "polygon"_s))
            m_shape = Shape::Poly;
        else {
            m_shape = Shape::Rect;
        }
        invalidateCachedRegion();
        break;
    case AttributeNames::coordsAttr:
        m_coords = parseHTMLListOfOfFloatingPointNumberValues(newValue.string());
        invalidateCachedRegion();
        break;
    case AttributeNames::altAttr:
        // Do nothing.
        break;
    default:
        HTMLAnchorElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
        break;
    }
}

void HTMLAreaElement::invalidateCachedRegion()
{
    m_lastSize = LayoutSize(-1, -1);
}

bool HTMLAreaElement::mapMouseEvent(LayoutPoint location, const LayoutSize& size, HitTestResult& result)
{
    if (m_lastSize != size) {
        m_region = makeUnique<Path>(getRegion(size));
        m_lastSize = size;
    }

    if (!m_region->contains(location, WindRule::EvenOdd))
        return false;
    
    result.setInnerNode(this);
    result.setURLElement(this);
    return true;
}

// FIXME: We should use RenderElement* instead of RenderObject* once we upstream iOS's DOMUIKitExtensions.{h, mm}.
Path HTMLAreaElement::computePath(RenderObject* obj) const
{
    if (!obj)
        return Path();
    
    // FIXME: This doesn't work correctly with transforms.
    FloatPoint absPos = obj->localToAbsolute();

    // Default should default to the size of the containing object.
    LayoutSize size = m_lastSize;
    if (isDefault())
        size = obj->absoluteOutlineBounds().size();
    
    Path p = getRegion(size);
    float zoomFactor = obj->style().usedZoom();
    if (zoomFactor != 1.0f) {
        AffineTransform zoomTransform;
        zoomTransform.scale(zoomFactor);
        p.transform(zoomTransform);
    }

    p.translate(toFloatSize(absPos));
    return p;
}

Path HTMLAreaElement::computePathForFocusRing(const LayoutSize& elementSize) const
{
    return getRegion(isDefault() ? elementSize : m_lastSize);
}

// FIXME: Use RenderElement* instead of RenderObject* once we upstream iOS's DOMUIKitExtensions.{h, mm}.
LayoutRect HTMLAreaElement::computeRect(RenderObject* obj) const
{
    return enclosingLayoutRect(computePath(obj).fastBoundingRect());
}

Path HTMLAreaElement::getRegion(const LayoutSize& size) const
{
    if (m_coords.isEmpty() && !isDefault())
        return Path();

    Path path;
    switch (m_shape) {
    case Shape::Poly:
        if (m_coords.size() >= 6) {
            int numPoints = m_coords.size() / 2;
            path.moveTo(FloatPoint(m_coords[0], m_coords[1]));
            for (int i = 1; i < numPoints; ++i)
                path.addLineTo(FloatPoint(m_coords[i * 2], m_coords[i * 2 + 1]));
            path.closeSubpath();
        }
        break;
    case Shape::Circle:
        if (m_coords.size() >= 3) {
            auto radius = m_coords[2];
            if (radius > 0)
                path.addEllipseInRect(FloatRect(m_coords[0] - radius, m_coords[1] - radius, 2 * radius, 2 * radius));
        }
        break;
    case Shape::Rect:
        if (m_coords.size() >= 4) {
            auto x0 = m_coords[0];
            auto y0 = m_coords[1];
            auto x1 = m_coords[2];
            auto y1 = m_coords[3];
            path.addRect(FloatRect(x0, y0, x1 - x0, y1 - y0));
        }
        break;
    case Shape::Default:
        path.addRect({ { 0, 0 }, size });
        break;
    }
    return path;
}

RefPtr<HTMLImageElement> HTMLAreaElement::imageElement() const
{
    if (RefPtr mapElement = dynamicDowncast<HTMLMapElement>(parentNode()))
        return mapElement->imageElement();
    return nullptr;
}

bool HTMLAreaElement::isKeyboardFocusable(KeyboardEvent*) const
{
    return isFocusable();
}
    
bool HTMLAreaElement::isMouseFocusable() const
{
    return isFocusable();
}

bool HTMLAreaElement::isFocusable() const
{
    RefPtr image = imageElement();
    if (!image || !image->hasFocusableStyle())
        return false;

    return supportsFocus() && tabIndexSetExplicitly().value_or(0) >= 0;
}
    
void HTMLAreaElement::setFocus(bool shouldBeFocused, FocusVisibility visibility)
{
    if (focused() == shouldBeFocused)
        return;

    HTMLAnchorElement::setFocus(shouldBeFocused, visibility);

    RefPtr imageElement = this->imageElement();
    if (!imageElement)
        return;

    if (CheckedPtr renderer = dynamicDowncast<RenderImage>(imageElement->renderer()))
        renderer->areaElementFocusChanged(this);
}

RefPtr<Element> HTMLAreaElement::focusAppearanceUpdateTarget()
{
    if (!isFocusable())
        return nullptr;
    return imageElement();
}
    
bool HTMLAreaElement::supportsFocus() const
{
    // If the AREA element was a link, it should support focus.
    // The inherited method is not used because it assumes that a render object must exist 
    // for the element to support focus. AREA elements do not have render objects.
    return isLink();
}

AtomString HTMLAreaElement::target() const
{
    return attributeWithoutSynchronization(targetAttr);
}

}
