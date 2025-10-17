/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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

#include "AnimationEffect.h"
#include "AnimationEffectPhase.h"
#include "RenderStyleConstants.h"
#include "Styleable.h"
#include "WebAnimation.h"
#include <wtf/Ref.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Animation;
class StyleOriginatedAnimationEvent;
class Element;
class RenderStyle;

class StyleOriginatedAnimation : public WebAnimation {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StyleOriginatedAnimation);
public:
    ~StyleOriginatedAnimation();

    bool isStyleOriginatedAnimation() const final { return true; }

    const std::optional<const Styleable> owningElement() const;
    const Animation& backingAnimation() const { return m_backingAnimation; }
    void setBackingAnimation(const Animation&);
    void cancelFromStyle(WebAnimation::Silently = WebAnimation::Silently::No);

    std::optional<WebAnimationTime> bindingsStartTime() const final;
    std::optional<WebAnimationTime> bindingsCurrentTime() const final;
    WebAnimation::PlayState bindingsPlayState() const final;
    WebAnimation::ReplaceState bindingsReplaceState() const final;
    bool bindingsPending() const final;
    WebAnimation::ReadyPromise& bindingsReady() final;
    WebAnimation::FinishedPromise& bindingsFinished() final;
    ExceptionOr<void> bindingsPlay() override;
    ExceptionOr<void> bindingsPause() override;

    void setTimeline(RefPtr<AnimationTimeline>&&) final;
    void cancel(WebAnimation::Silently = WebAnimation::Silently::No) final;

    void tick() override;

    bool canHaveGlobalPosition() final;

    void flushPendingStyleChanges() const;

protected:
    StyleOriginatedAnimation(const Styleable&, const Animation&);

    void initialize(const RenderStyle* oldStyle, const RenderStyle& newStyle, const Style::ResolutionContext&);
    virtual void syncPropertiesWithBackingAnimation();
    virtual Ref<StyleOriginatedAnimationEvent> createEvent(const AtomString& eventType, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>&) = 0;

private:
    void disassociateFromOwningElement();
    AnimationEffectPhase phaseWithoutEffect() const;
    enum class ShouldFireEvents : uint8_t { No, YesForCSSAnimation, YesForCSSTransition };
    ShouldFireEvents shouldFireDOMEvents() const;
    void invalidateDOMEvents(ShouldFireEvents, WebAnimationTime elapsedTime = 0_s);
    void enqueueDOMEvent(const AtomString&, WebAnimationTime elapsedTime, WebAnimationTime scheduledEffectTime);

    WebAnimationTime effectTimeAtStart() const;
    WebAnimationTime effectTimeAtIteration(double) const;
    WebAnimationTime effectTimeAtEnd() const;

    bool m_wasPending { false };
    AnimationEffectPhase m_previousPhase { AnimationEffectPhase::Idle };

    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_owningElement;
    std::optional<Style::PseudoElementIdentifier> m_owningPseudoElementIdentifier;
    Ref<Animation> m_backingAnimation;
    double m_previousIteration;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_WEB_ANIMATION(StyleOriginatedAnimation, isStyleOriginatedAnimation())
