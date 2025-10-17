/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
#import "config.h"
#import "RemoteAcceleratedEffectStack.h"

#if ENABLE(THREADED_ANIMATION_RESOLUTION)

#import <WebCore/AcceleratedEffectStack.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RemoteAcceleratedEffectStack);

Ref<RemoteAcceleratedEffectStack> RemoteAcceleratedEffectStack::create(WebCore::FloatRect bounds, Seconds acceleratedTimelineTimeOrigin)
{
    return adoptRef(*new RemoteAcceleratedEffectStack(bounds, acceleratedTimelineTimeOrigin));
}

RemoteAcceleratedEffectStack::RemoteAcceleratedEffectStack(WebCore::FloatRect bounds, Seconds acceleratedTimelineTimeOrigin)
    : m_bounds(bounds)
    , m_acceleratedTimelineTimeOrigin(acceleratedTimelineTimeOrigin)
{
}

void RemoteAcceleratedEffectStack::setEffects(WebCore::AcceleratedEffects&& effects)
{
    WebCore::AcceleratedEffectStack::setEffects(WTFMove(effects));

    bool affectsFilter = false;
    bool affectsOpacity = false;
    bool affectsTransform = false;

    for (auto& effect : m_backdropLayerEffects.isEmpty() ? m_primaryLayerEffects : m_backdropLayerEffects) {
        auto& properties = effect->animatedProperties();
        affectsFilter = affectsFilter || properties.containsAny({ WebCore::AcceleratedEffectProperty::Filter, WebCore::AcceleratedEffectProperty::BackdropFilter });
        affectsOpacity = affectsOpacity || properties.contains(WebCore::AcceleratedEffectProperty::Opacity);
        affectsTransform = affectsTransform || properties.containsAny(WebCore::transformRelatedAcceleratedProperties);
        if (affectsFilter && affectsOpacity && affectsTransform)
            break;
    }

    ASSERT(affectsFilter || affectsOpacity || affectsTransform);

    if (affectsFilter)
        m_affectedLayerProperties.add(LayerProperty::Filter);
    if (affectsOpacity)
        m_affectedLayerProperties.add(LayerProperty::Opacity);
    if (affectsTransform)
        m_affectedLayerProperties.add(LayerProperty::Transform);
}

#if PLATFORM(MAC)
const WebCore::FilterOperations* RemoteAcceleratedEffectStack::longestFilterList() const
{
    if (!m_affectedLayerProperties.contains(LayerProperty::Filter))
        return nullptr;

    auto isBackdrop = !m_backdropLayerEffects.isEmpty();
    auto filterProperty = isBackdrop ? WebCore::AcceleratedEffectProperty::BackdropFilter : WebCore::AcceleratedEffectProperty::Filter;
    auto& effects = isBackdrop ? m_backdropLayerEffects : m_primaryLayerEffects;

    const WebCore::FilterOperations* longestFilterList = nullptr;
    for (auto& effect : effects) {
        if (!effect->animatedProperties().contains(filterProperty))
            continue;
        for (auto& keyframe : effect->keyframes()) {
            if (!keyframe.animatedProperties().contains(filterProperty))
                continue;
            auto& filter = isBackdrop ? keyframe.values().backdropFilter : keyframe.values().filter;
            if (!longestFilterList || longestFilterList->size() < filter.size())
                longestFilterList = &filter;
        }
    }

    if (longestFilterList) {
        auto& baseFilter = isBackdrop ? m_baseValues.backdropFilter : m_baseValues.filter;
        if (longestFilterList->size() < baseFilter.size())
            longestFilterList = &baseFilter;
    }

    return longestFilterList && !longestFilterList->isEmpty() ? longestFilterList : nullptr;
}

void RemoteAcceleratedEffectStack::initEffectsFromMainThread(PlatformLayer *layer, MonotonicTime now)
{
    ASSERT(m_filterPresentationModifiers.isEmpty());
    ASSERT(!m_opacityPresentationModifier);
    ASSERT(!m_transformPresentationModifier);
    ASSERT(!m_presentationModifierGroup);

    auto computedValues = computeValues(now);

    auto* canonicalFilters = longestFilterList();

    auto numberOfPresentationModifiers = [&]() {
        size_t count = 0;
        if (m_affectedLayerProperties.contains(LayerProperty::Filter)) {
            ASSERT(canonicalFilters);
            count += WebCore::PlatformCAFilters::presentationModifierCount(*canonicalFilters);
        }
        if (m_affectedLayerProperties.contains(LayerProperty::Opacity))
            count++;
        if (m_affectedLayerProperties.contains(LayerProperty::Transform))
            count++;
        return count;
    }();

    m_presentationModifierGroup = [CAPresentationModifierGroup groupWithCapacity:numberOfPresentationModifiers];

    if (m_affectedLayerProperties.contains(LayerProperty::Filter)) {
        WebCore::PlatformCAFilters::presentationModifiers(computedValues.filter, longestFilterList(), m_filterPresentationModifiers, m_presentationModifierGroup);
        for (auto& filterPresentationModifier : m_filterPresentationModifiers)
            [layer addPresentationModifier:filterPresentationModifier.second.get()];
    }

    if (m_affectedLayerProperties.contains(LayerProperty::Opacity)) {
        auto *opacity = @(computedValues.opacity);
        m_opacityPresentationModifier = adoptNS([[CAPresentationModifier alloc] initWithKeyPath:@"opacity" initialValue:opacity additive:NO group:m_presentationModifierGroup.get()]);
        [layer addPresentationModifier:m_opacityPresentationModifier.get()];
    }

    if (m_affectedLayerProperties.contains(LayerProperty::Transform)) {
        auto computedTransform = computedValues.computedTransformationMatrix(m_bounds);
        auto *transform = [NSValue valueWithCATransform3D:computedTransform];
        m_transformPresentationModifier = adoptNS([[CAPresentationModifier alloc] initWithKeyPath:@"transform" initialValue:transform additive:NO group:m_presentationModifierGroup.get()]);
        [layer addPresentationModifier:m_transformPresentationModifier.get()];
    }

    [m_presentationModifierGroup flushWithTransaction];
}

void RemoteAcceleratedEffectStack::applyEffectsFromScrollingThread(MonotonicTime now) const
{
    ASSERT(m_presentationModifierGroup);

    auto computedValues = computeValues(now);

    if (!m_filterPresentationModifiers.isEmpty())
        WebCore::PlatformCAFilters::updatePresentationModifiers(computedValues.filter, m_filterPresentationModifiers);

    if (m_opacityPresentationModifier) {
        auto *opacity = @(computedValues.opacity);
        [m_opacityPresentationModifier setValue:opacity];
    }

    if (m_transformPresentationModifier) {
        auto computedTransform = computedValues.computedTransformationMatrix(m_bounds);
        auto *transform = [NSValue valueWithCATransform3D:computedTransform];
        [m_transformPresentationModifier setValue:transform];
    }

    [m_presentationModifierGroup flush];
}
#endif

void RemoteAcceleratedEffectStack::applyEffectsFromMainThread(PlatformLayer *layer, MonotonicTime now, bool backdropRootIsOpaque) const
{
    auto computedValues = computeValues(now);

    if (m_affectedLayerProperties.contains(LayerProperty::Filter))
        WebCore::PlatformCAFilters::setFiltersOnLayer(layer, computedValues.filter, backdropRootIsOpaque);

    if (m_affectedLayerProperties.contains(LayerProperty::Opacity))
        [layer setOpacity:computedValues.opacity];

    if (m_affectedLayerProperties.contains(LayerProperty::Transform)) {
        auto computedTransform = computedValues.computedTransformationMatrix(m_bounds);
        [layer setTransform:computedTransform];
    }
}

WebCore::AcceleratedEffectValues RemoteAcceleratedEffectStack::computeValues(MonotonicTime now) const
{
    auto values = m_baseValues;
    auto currentTime = now.secondsSinceEpoch() - m_acceleratedTimelineTimeOrigin;
    for (auto& effect : m_backdropLayerEffects.isEmpty() ? m_primaryLayerEffects : m_backdropLayerEffects)
        effect->apply(currentTime, values, m_bounds);
    return values;
}

void RemoteAcceleratedEffectStack::clear(PlatformLayer *layer)
{
#if PLATFORM(MAC)
    ASSERT(m_presentationModifierGroup);

    for (auto& filterPresentationModifier : m_filterPresentationModifiers)
        [layer removePresentationModifier:filterPresentationModifier.second.get()];
    if (m_opacityPresentationModifier)
        [layer removePresentationModifier:m_opacityPresentationModifier.get()];
    if (m_transformPresentationModifier)
        [layer removePresentationModifier:m_transformPresentationModifier.get()];

    [m_presentationModifierGroup flushWithTransaction];

    m_filterPresentationModifiers.clear();
    m_opacityPresentationModifier = nil;
    m_transformPresentationModifier = nil;
    m_presentationModifierGroup = nil;
#endif
}

} // namespace WebKit

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)
