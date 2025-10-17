/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
#include "AnimationTimeline.h"

#include "AnimationTimelinesController.h"
#include "KeyframeEffect.h"
#include "KeyframeEffectStack.h"
#include "StyleResolver.h"
#include "Styleable.h"
#include "WebAnimationUtilities.h"

namespace WebCore {

AnimationTimeline::AnimationTimeline(std::optional<WebAnimationTime> duration)
    : m_duration(duration)
{
}

AnimationTimeline::~AnimationTimeline() = default;

void AnimationTimeline::animationTimingDidChange(WebAnimation& animation)
{
    updateGlobalPosition(animation);

    if (m_animations.add(animation)) {
        auto* timeline = animation.timeline();
        if (timeline && timeline != this)
            timeline->removeAnimation(animation);
        else if (timeline == this) {
            if (auto* keyframeEffect = dynamicDowncast<KeyframeEffect>(animation.effect())) {
                if (auto styleable = keyframeEffect->targetStyleable())
                    styleable->animationWasAdded(animation);
            }
        }
    }
}

void AnimationTimeline::updateGlobalPosition(WebAnimation& animation)
{
    static uint64_t s_globalPosition = 0;
    if (!animation.globalPosition() && animation.canHaveGlobalPosition())
        animation.setGlobalPosition(++s_globalPosition);
}

void AnimationTimeline::removeAnimation(WebAnimation& animation)
{
    ASSERT(!animation.timeline() || animation.timeline() == this);
    m_animations.remove(animation);
    if (auto* keyframeEffect = dynamicDowncast<KeyframeEffect>(animation.effect())) {
        if (auto styleable = keyframeEffect->targetStyleable()) {
            styleable->animationWasRemoved(animation);
            styleable->ensureKeyframeEffectStack().removeEffect(*keyframeEffect);
        }
    }
}

void AnimationTimeline::detachFromDocument()
{
    if (CheckedPtr controller = this->controller())
        controller->removeTimeline(*this);

    auto& animationsToRemove = m_animations;
    while (!animationsToRemove.isEmpty())
        animationsToRemove.first()->remove();
}

void AnimationTimeline::suspendAnimations()
{
    for (const auto& animation : m_animations)
        animation->setSuspended(true);
}

void AnimationTimeline::resumeAnimations()
{
    for (const auto& animation : m_animations)
        animation->setSuspended(false);
}

bool AnimationTimeline::animationsAreSuspended() const
{
    return controller() && controller()->animationsAreSuspended();
}

} // namespace WebCore
