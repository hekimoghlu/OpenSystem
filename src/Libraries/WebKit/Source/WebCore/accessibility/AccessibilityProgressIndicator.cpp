/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#include "AccessibilityProgressIndicator.h"

#include "AXObjectCache.h"
#include "FloatConversion.h"
#include "HTMLMeterElement.h"
#include "HTMLNames.h"
#include "HTMLProgressElement.h"
#include "LocalizedStrings.h"
#include "RenderMeter.h"
#include "RenderObject.h"
#include "RenderProgress.h"

namespace WebCore {
    
using namespace HTMLNames;

AccessibilityProgressIndicator::AccessibilityProgressIndicator(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
    ASSERT(is<RenderProgress>(renderer) || is<RenderMeter>(renderer) || is<HTMLProgressElement>(renderer.node()) || is<HTMLMeterElement>(renderer.node()));
}

Ref<AccessibilityProgressIndicator> AccessibilityProgressIndicator::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityProgressIndicator(axID, renderer));
}

bool AccessibilityProgressIndicator::computeIsIgnored() const
{
    return isIgnoredByDefault();
}
    
String AccessibilityProgressIndicator::valueDescription() const
{
    // If the author has explicitly provided a value through aria-valuetext, use it.
    String description = AccessibilityRenderObject::valueDescription();
    if (!description.isEmpty())
        return description;

    RefPtr meter = meterElement();
    if (!meter)
        return description;

    // The HTML spec encourages authors to include a textual representation of the meter's state in
    // the element's contents. We'll fall back on that if there is not a more accessible alternative.
    if (auto* nodeObject = dynamicDowncast<AccessibilityNodeObject>(axObjectCache()->getOrCreate(*meter)))
        description = nodeObject->accessibilityDescriptionForChildren();

    if (description.isEmpty())
        description = meter->textContent();

    String gaugeRegionValue = gaugeRegionValueDescription();
    if (!gaugeRegionValue.isEmpty())
        description = description.isEmpty() ? gaugeRegionValue : makeString(description, ", "_s,  gaugeRegionValue);

    return description;
}

bool AccessibilityProgressIndicator::isIndeterminate() const
{
    if (auto* progress = progressElement())
        return !progress->hasAttribute(valueAttr);
    return false;
}

float AccessibilityProgressIndicator::valueForRange() const
{
    if (auto* progress = progressElement(); progress && progress->position() >= 0)
        return narrowPrecisionToFloat(progress->value());

    if (auto* meter = meterElement())
        return narrowPrecisionToFloat(meter->value());

    // Indeterminate progress bar should return 0.
    return 0.0;
}

float AccessibilityProgressIndicator::maxValueForRange() const
{
    if (auto* progress = progressElement())
        return narrowPrecisionToFloat(progress->max());

    if (auto* meter = meterElement())
        return narrowPrecisionToFloat(meter->max());

    return 0.0;
}

float AccessibilityProgressIndicator::minValueForRange() const
{
    if (progressElement())
        return 0.0;

    if (auto* meter = meterElement())
        return narrowPrecisionToFloat(meter->min());

    return 0.0;
}

AccessibilityRole AccessibilityProgressIndicator::determineAccessibilityRole()
{
    if (meterElement())
        return AccessibilityRole::Meter;
    return AccessibilityRole::ProgressIndicator;
}

HTMLProgressElement* AccessibilityProgressIndicator::progressElement() const
{
    return dynamicDowncast<HTMLProgressElement>(node());
}

HTMLMeterElement* AccessibilityProgressIndicator::meterElement() const
{
    return dynamicDowncast<HTMLMeterElement>(node());
}

String AccessibilityProgressIndicator::gaugeRegionValueDescription() const
{
#if PLATFORM(COCOA)
    auto* meterElement = this->meterElement();
    if (!meterElement)
        return String();

    // Only expose this when the author has explicitly specified the following attributes.
    if (!hasAttribute(lowAttr) && !hasAttribute(highAttr) && !hasAttribute(optimumAttr))
        return String();
    
    switch (meterElement->gaugeRegion()) {
    case HTMLMeterElement::GaugeRegionOptimum:
        return AXMeterGaugeRegionOptimumText();
    case HTMLMeterElement::GaugeRegionSuboptimal:
        return AXMeterGaugeRegionSuboptimalText();
    case HTMLMeterElement::GaugeRegionEvenLessGood:
        return AXMeterGaugeRegionLessGoodText();
    }
#endif
    return String();
}

} // namespace WebCore

