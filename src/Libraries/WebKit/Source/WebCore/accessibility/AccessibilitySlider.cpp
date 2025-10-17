/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#include "AccessibilitySlider.h"

#include "AXObjectCache.h"
#include "HTMLInputElement.h"
#include "HTMLNames.h"
#include "RenderSlider.h"
#include "RenderStyleInlines.h"
#include "SliderThumbElement.h"
#include "StyleAppearance.h"
#include <wtf/Scope.h>

namespace WebCore {
    
using namespace HTMLNames;

AccessibilitySlider::AccessibilitySlider(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

Ref<AccessibilitySlider> AccessibilitySlider::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilitySlider(axID, renderer));
}

AccessibilityOrientation AccessibilitySlider::orientation() const
{
    auto ariaOrientation = getAttribute(aria_orientationAttr);
    if (equalLettersIgnoringASCIICase(ariaOrientation, "horizontal"_s))
        return AccessibilityOrientation::Horizontal;
    if (equalLettersIgnoringASCIICase(ariaOrientation, "vertical"_s))
        return AccessibilityOrientation::Vertical;
    if (equalLettersIgnoringASCIICase(ariaOrientation, "undefined"_s))
        return AccessibilityOrientation::Undefined;

    const auto* style = this->style();
    // Default to horizontal in the unknown case.
    if (!style)
        return AccessibilityOrientation::Horizontal;

    auto styleAppearance = style->usedAppearance();
    switch (styleAppearance) {
    case StyleAppearance::SliderThumbHorizontal:
    case StyleAppearance::SliderHorizontal:
        return AccessibilityOrientation::Horizontal;
    
    case StyleAppearance::SliderThumbVertical:
    case StyleAppearance::SliderVertical:
        return AccessibilityOrientation::Vertical;
        
    default:
        return AccessibilityOrientation::Horizontal;
    }
}
    
void AccessibilitySlider::addChildren()
{
    ASSERT(!m_childrenInitialized); 
    m_childrenInitialized = true;
    auto clearDirtySubtree = makeScopeExit([&] {
        m_subtreeDirty = false;
    });

    auto* cache = axObjectCache();
    if (!cache)
        return;

    Ref thumb = downcast<AccessibilitySliderThumb>(*cache->create(AccessibilityRole::SliderThumb));
    thumb->setParent(this);

    // Before actually adding the value indicator to the hierarchy,
    // allow the platform to make a final decision about it.
    if (thumb->isIgnored())
        cache->remove(thumb->objectID());
    else
        addChild(thumb.get());
}

AccessibilityObject* AccessibilitySlider::elementAccessibilityHitTest(const IntPoint& point) const
{
    if (m_children.size()) {
        ASSERT(m_children.size() == 1);
        if (m_children[0]->elementRect().contains(point))
            return dynamicDowncast<AccessibilityObject>(m_children[0].get());
    }
    
    return axObjectCache()->getOrCreate(renderer());
}

float AccessibilitySlider::valueForRange() const
{
    if (auto* input = inputElement())
        return input->value().toFloat();
    return 0;
}

float AccessibilitySlider::maxValueForRange() const
{
    if (auto* input = inputElement())
        return static_cast<float>(input->maximum());
    return 0;
}

float AccessibilitySlider::minValueForRange() const
{
    if (auto* input = inputElement())
        return static_cast<float>(input->minimum());
    return 0;
}

bool AccessibilitySlider::setValue(const String& value)
{
    RefPtr input = inputElement();
    if (!input)
        return false;

    if (input->value() != value)
        input->setValue(value, DispatchInputAndChangeEvent);
    return true;
}

HTMLInputElement* AccessibilitySlider::inputElement() const
{
    return dynamicDowncast<HTMLInputElement>(node());
}


AccessibilitySliderThumb::AccessibilitySliderThumb(AXID axID)
    : AccessibilityMockObject(axID)
{
}

Ref<AccessibilitySliderThumb> AccessibilitySliderThumb::create(AXID axID)
{
    return adoptRef(*new AccessibilitySliderThumb(axID));
}
    
LayoutRect AccessibilitySliderThumb::elementRect() const
{
    if (!m_parent)
        return LayoutRect();
    
    auto* sliderRenderer = dynamicDowncast<RenderSlider>(m_parent->renderer());
    if (!sliderRenderer)
        return LayoutRect();
    if (auto* thumbRenderer = sliderRenderer->element().sliderThumbElement()->renderer())
        return thumbRenderer->absoluteBoundingBoxRect();
    return LayoutRect();
}

bool AccessibilitySliderThumb::computeIsIgnored() const
{
    return isIgnoredByDefault();
}

} // namespace WebCore
