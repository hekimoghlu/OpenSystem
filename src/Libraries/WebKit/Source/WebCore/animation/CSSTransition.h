/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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

#include "CSSPropertyNames.h"
#include "StyleOriginatedAnimation.h"
#include "Styleable.h"
#include "WebAnimationTypes.h"
#include <wtf/Markable.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Ref.h>
#include <wtf/Seconds.h>

namespace WebCore {

class Animation;
class RenderStyle;

class CSSTransition final : public StyleOriginatedAnimation {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSTransition);
public:
    static Ref<CSSTransition> create(const Styleable&, const AnimatableCSSProperty&, MonotonicTime generationTime, const Animation&, const RenderStyle& oldStyle, const RenderStyle& newStyle, Seconds delay, Seconds duration, const RenderStyle& reversingAdjustedStartStyle, double);

    virtual ~CSSTransition();

    const AtomString transitionProperty() const;
    AnimatableCSSProperty property() const { return m_property; }
    MonotonicTime generationTime() const { return m_generationTime; }
    std::optional<Seconds> timelineTimeAtCreation() const { return m_timelineTimeAtCreation; }
    const RenderStyle& targetStyle() const { return *m_targetStyle; }
    const RenderStyle& currentStyle() const { return *m_currentStyle; }
    const RenderStyle& reversingAdjustedStartStyle() const { return *m_reversingAdjustedStartStyle; }
    double reversingShorteningFactor() const { return m_reversingShorteningFactor; }

private:
    CSSTransition(const Styleable&, const AnimatableCSSProperty&, MonotonicTime generationTime, const Animation&, const RenderStyle& oldStyle, const RenderStyle& targetStyle, const RenderStyle& reversingAdjustedStartStyle, double);
    void setTimingProperties(Seconds delay, Seconds duration);
    Ref<StyleOriginatedAnimationEvent> createEvent(const AtomString& eventType, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>&) final;
    OptionSet<AnimationImpact> resolve(RenderStyle& targetStyle, const Style::ResolutionContext&, std::optional<Seconds>) final;
    void animationDidFinish() final;
    bool isCSSTransition() const final { return true; }

    AnimatableCSSProperty m_property;
    MonotonicTime m_generationTime;
    Markable<Seconds, Seconds::MarkableTraits> m_timelineTimeAtCreation;
    std::unique_ptr<RenderStyle> m_targetStyle;
    std::unique_ptr<RenderStyle> m_currentStyle;
    std::unique_ptr<RenderStyle> m_reversingAdjustedStartStyle;
    double m_reversingShorteningFactor;

};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_WEB_ANIMATION(CSSTransition, isCSSTransition())
