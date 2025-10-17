/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "WillChangeData.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WillChangeData);

bool WillChangeData::operator==(const WillChangeData& other) const
{
    return m_animatableFeatures == other.m_animatableFeatures;
}

bool WillChangeData::containsScrollPosition() const
{
    for (const auto& feature : m_animatableFeatures) {
        if (feature.feature() == Feature::ScrollPosition)
            return true;
    }
    return false;
}

bool WillChangeData::containsContents() const
{
    for (const auto& feature : m_animatableFeatures) {
        if (feature.feature() == Feature::Contents)
            return true;
    }
    return false;
}

bool WillChangeData::containsProperty(CSSPropertyID property) const
{
    for (const auto& feature : m_animatableFeatures) {
        if (feature.property() == property)
            return true;
    }
    return false;
}

bool WillChangeData::createsContainingBlockForAbsolutelyPositioned(bool isRootElement) const
{
    return createsContainingBlockForOutOfFlowPositioned(isRootElement)
        || containsProperty(CSSPropertyPosition);
}

bool WillChangeData::createsContainingBlockForOutOfFlowPositioned(bool isRootElement) const
{
    return containsProperty(CSSPropertyPerspective)
        // CSS transforms
        || containsProperty(CSSPropertyTransform)
        || containsProperty(CSSPropertyTransformStyle)
        || containsProperty(CSSPropertyTranslate)
        || containsProperty(CSSPropertyRotate)
        || containsProperty(CSSPropertyScale)
        || containsProperty(CSSPropertyContain)
        // CSS filter & backdrop-filter
        // FIXME: exclude root element for those properties (bug 225034)
        || (containsProperty(CSSPropertyBackdropFilter) && !isRootElement)
        || (containsProperty(CSSPropertyWebkitBackdropFilter) && !isRootElement)
        || containsProperty(CSSPropertyFilter);
}

bool WillChangeData::canBeBackdropRoot() const
{
    return containsProperty(CSSPropertyOpacity)
        || containsProperty(CSSPropertyBackdropFilter)
        || containsProperty(CSSPropertyWebkitBackdropFilter)
        || containsProperty(CSSPropertyClipPath)
        || containsProperty(CSSPropertyFilter)
        || containsProperty(CSSPropertyMixBlendMode)
        || containsProperty(CSSPropertyMask);
}

// "If any non-initial value of a property would create a stacking context on the element,
// specifying that property in will-change must create a stacking context on the element."
bool WillChangeData::propertyCreatesStackingContext(CSSPropertyID property)
{
    switch (property) {
    case CSSPropertyPerspective:
    case CSSPropertyWebkitPerspective:
    case CSSPropertyScale:
    case CSSPropertyRotate:
    case CSSPropertyTranslate:
    case CSSPropertyTransform:
    case CSSPropertyTransformStyle:
    case CSSPropertyClipPath:
    case CSSPropertyMask:
    case CSSPropertyWebkitMask:
    case CSSPropertyOpacity:
    case CSSPropertyPosition:
    case CSSPropertyZIndex:
    case CSSPropertyWebkitBoxReflect:
    case CSSPropertyMixBlendMode:
    case CSSPropertyIsolation:
    case CSSPropertyFilter:
    case CSSPropertyBackdropFilter:
    case CSSPropertyWebkitBackdropFilter:
    case CSSPropertyMaskImage:
    case CSSPropertyMaskBorder:
    case CSSPropertyWebkitMaskBoxImage:
#if ENABLE(OVERFLOW_SCROLLING_TOUCH)
    case CSSPropertyWebkitOverflowScrolling:
#endif
    case CSSPropertyContain:
        return true;
    default:
        return false;
    }
}

static bool propertyTriggersCompositing(CSSPropertyID property)
{
    switch (property) {
    case CSSPropertyOpacity:
    case CSSPropertyFilter:
    case CSSPropertyBackdropFilter:
    case CSSPropertyWebkitBackdropFilter:
        return true;
    default:
        return false;
    }
}

static bool propertyTriggersCompositingOnBoxesOnly(CSSPropertyID property)
{
    // Don't trigger for perspective and transform-style, because those
    // only do compositing if they have a 3d-transformed descendant and
    // we don't want to do compositing all the time.
    // Similarly, we don't want -webkit-overflow-scrolling-touch to
    // always composite if there's no scrollable overflow.
    switch (property) {
    case CSSPropertyScale:
    case CSSPropertyRotate:
    case CSSPropertyTranslate:
    case CSSPropertyTransform:
        return true;
    default:
        return false;
    }
}

void WillChangeData::addFeature(Feature feature, CSSPropertyID propertyID)
{
    ASSERT(feature == Feature::Property || propertyID == CSSPropertyInvalid);
    m_animatableFeatures.append(AnimatableFeature(feature, propertyID));

    m_canCreateStackingContext |= propertyCreatesStackingContext(propertyID);

    m_canTriggerCompositingOnInline |= propertyTriggersCompositing(propertyID);
    m_canTriggerCompositing |= m_canTriggerCompositingOnInline | propertyTriggersCompositingOnBoxesOnly(propertyID);
}

WillChangeData::FeaturePropertyPair WillChangeData::featureAt(size_t index) const
{
    if (index >= m_animatableFeatures.size())
        return FeaturePropertyPair(Feature::Invalid, CSSPropertyInvalid);

    return m_animatableFeatures[index].featurePropertyPair();
}

} // namespace WebCore
