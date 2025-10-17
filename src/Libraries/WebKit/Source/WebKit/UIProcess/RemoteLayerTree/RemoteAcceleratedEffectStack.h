/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#pragma once

#if ENABLE(THREADED_ANIMATION_RESOLUTION)

#include <WebCore/AcceleratedEffect.h>
#include <WebCore/AcceleratedEffectStack.h>
#include <WebCore/AcceleratedEffectValues.h>
#include <WebCore/PlatformCAFilters.h>
#include <WebCore/PlatformLayer.h>
#include <wtf/OptionSet.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS CAPresentationModifierGroup;
OBJC_CLASS CAPresentationModifier;

namespace WebKit {

class RemoteAcceleratedEffectStack final : public WebCore::AcceleratedEffectStack {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RemoteAcceleratedEffectStack);
public:
    static Ref<RemoteAcceleratedEffectStack> create(WebCore::FloatRect, Seconds);

    void setEffects(WebCore::AcceleratedEffects&&) final;

#if PLATFORM(MAC)
    void initEffectsFromMainThread(PlatformLayer*, MonotonicTime now);
    void applyEffectsFromScrollingThread(MonotonicTime now) const;
#endif

    void applyEffectsFromMainThread(PlatformLayer*, MonotonicTime now, bool backdropRootIsOpaque) const;

    void clear(PlatformLayer*);

private:
    explicit RemoteAcceleratedEffectStack(WebCore::FloatRect, Seconds);

    WebCore::AcceleratedEffectValues computeValues(MonotonicTime now) const;

#if PLATFORM(MAC)
    const WebCore::FilterOperations* longestFilterList() const;
#endif

    enum class LayerProperty : uint8_t {
        Opacity = 1 << 1,
        Transform = 1 << 2,
        Filter = 1 << 3
    };

    OptionSet<LayerProperty> m_affectedLayerProperties;

    WebCore::FloatRect m_bounds;
    Seconds m_acceleratedTimelineTimeOrigin;

#if PLATFORM(MAC)
    RetainPtr<CAPresentationModifierGroup> m_presentationModifierGroup;
    RetainPtr<CAPresentationModifier> m_opacityPresentationModifier;
    RetainPtr<CAPresentationModifier> m_transformPresentationModifier;
    Vector<WebCore::TypedFilterPresentationModifier> m_filterPresentationModifiers;
#endif
};

} // namespace WebKit

#endif // ENABLE(THREADED_ANIMATION_RESOLUTION)
