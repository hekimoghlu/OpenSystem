/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#include "TimelineRange.h"
#include "WebAnimationTypes.h"
#include <wtf/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class AnimationTimelinesController;
class WebAnimation;

class AnimationTimeline : public RefCountedAndCanMakeWeakPtr<AnimationTimeline> {
public:
    virtual ~AnimationTimeline();

    virtual bool isDocumentTimeline() const { return false; }
    virtual bool isScrollTimeline() const { return false; }
    virtual bool isViewTimeline() const { return false; }

    bool isMonotonic() const { return !m_duration; }
    bool isProgressBased() const { return !isMonotonic(); }

    const AnimationCollection& relevantAnimations() const { return m_animations; }

    virtual void animationTimingDidChange(WebAnimation&);
    virtual void removeAnimation(WebAnimation&);

    virtual std::optional<WebAnimationTime> currentTime() { return m_currentTime; }
    virtual std::optional<WebAnimationTime> duration() const { return m_duration; }

    virtual void detachFromDocument();

    enum class ShouldUpdateAnimationsAndSendEvents : bool { No, Yes };
    virtual ShouldUpdateAnimationsAndSendEvents documentWillUpdateAnimationsAndSendEvents() { return ShouldUpdateAnimationsAndSendEvents::No; }

    virtual void suspendAnimations();
    virtual void resumeAnimations();
    bool animationsAreSuspended() const;

    virtual AnimationTimelinesController* controller() const { return nullptr; }

    virtual TimelineRange defaultRange() const { return { }; }
    static void updateGlobalPosition(WebAnimation&);
protected:
    AnimationTimeline(std::optional<WebAnimationTime> = std::nullopt);

    AnimationCollection m_animations;

private:

    std::optional<WebAnimationTime> m_currentTime;
    std::optional<WebAnimationTime> m_duration;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_ANIMATION_TIMELINE(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
static bool isType(const WebCore::AnimationTimeline& value) { return value.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()
