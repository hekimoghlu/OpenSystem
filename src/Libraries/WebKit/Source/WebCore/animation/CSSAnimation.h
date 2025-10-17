/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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

#include "StyleOriginatedAnimation.h"
#include "Styleable.h"
#include <wtf/OptionSet.h>
#include <wtf/Ref.h>

namespace WebCore {

class Animation;
class RenderStyle;

class CSSAnimation final : public StyleOriginatedAnimation {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSAnimation);
public:
    static Ref<CSSAnimation> create(const Styleable&, const Animation&, const RenderStyle* oldStyle, const RenderStyle& newStyle, const Style::ResolutionContext&);
    ~CSSAnimation() = default;

    bool isCSSAnimation() const override { return true; }
    const String& animationName() const { return m_animationName; }

    void effectTimingWasUpdatedUsingBindings(OptionalEffectTiming);
    void effectKeyframesWereSetUsingBindings();
    void effectCompositeOperationWasSetUsingBindings();
    void keyframesRuleDidChange();
    void updateKeyframesIfNeeded(const RenderStyle* oldStyle, const RenderStyle& newStyle, const Style::ResolutionContext&);

    void syncStyleOriginatedTimeline();

private:
    CSSAnimation(const Styleable&, const Animation&);

    void syncPropertiesWithBackingAnimation() final;
    Ref<StyleOriginatedAnimationEvent> createEvent(const AtomString& eventType, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>&) final;

    AnimationTimeline* bindingsTimeline() const final;
    void setBindingsTimeline(RefPtr<AnimationTimeline>&&) final;
    ExceptionOr<void> bindingsPlay() final;
    ExceptionOr<void> bindingsPause() final;
    void setBindingsEffect(RefPtr<AnimationEffect>&&) final;
    ExceptionOr<void> setBindingsStartTime(const std::optional<WebAnimationTime>&) final;
    ExceptionOr<void> bindingsReverse() final;
    void setBindingsRangeStart(TimelineRangeValue&&) final;
    void setBindingsRangeEnd(TimelineRangeValue&&) final;

    enum class Property : uint16_t {
        Name = 1 << 0,
        Duration = 1 << 1,
        TimingFunction = 1 << 2,
        IterationCount = 1 << 3,
        Direction = 1 << 4,
        PlayState = 1 << 5,
        Delay = 1 << 6,
        FillMode = 1 << 7,
        Keyframes = 1 << 8,
        CompositeOperation = 1 << 9,
        Timeline = 1 << 10,
        RangeStart = 1 << 11,
        RangeEnd = 1 << 12,
    };

    String m_animationName;
    OptionSet<Property> m_overriddenProperties;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_WEB_ANIMATION(CSSAnimation, isCSSAnimation())
