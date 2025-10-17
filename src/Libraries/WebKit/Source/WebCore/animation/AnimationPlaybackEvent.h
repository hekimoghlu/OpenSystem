/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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

#include "AnimationEventBase.h"
#include "AnimationPlaybackEventInit.h"
#include "WebAnimationTypes.h"
#include <wtf/Markable.h>

namespace WebCore {

class AnimationPlaybackEvent final : public AnimationEventBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AnimationPlaybackEvent);
public:
    static Ref<AnimationPlaybackEvent> create(const AtomString& type, WebAnimation* animation, std::optional<WebAnimationTime> scheduledTime, std::optional<WebAnimationTime> timelineTime, std::optional<WebAnimationTime> currentTime)
    {
        return adoptRef(*new AnimationPlaybackEvent(type, animation, scheduledTime, timelineTime, currentTime));
    }

    static Ref<AnimationPlaybackEvent> create(const AtomString& type, const AnimationPlaybackEventInit& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new AnimationPlaybackEvent(type, initializer, isTrusted));
    }

    virtual ~AnimationPlaybackEvent();

    bool isAnimationPlaybackEvent() const final { return true; }

    std::optional<WebAnimationTime> timelineTime() const { return m_timelineTime; }
    std::optional<WebAnimationTime> currentTime() const { return m_currentTime; }

private:
    AnimationPlaybackEvent(const AtomString&, WebAnimation*, std::optional<WebAnimationTime> scheduledTime, std::optional<WebAnimationTime> timelineTime, std::optional<WebAnimationTime> currentTime);
    AnimationPlaybackEvent(const AtomString&, const AnimationPlaybackEventInit&, IsTrusted);

    std::optional<WebAnimationTime> m_timelineTime;
    std::optional<WebAnimationTime> m_currentTime;
};

}

SPECIALIZE_TYPE_TRAITS_ANIMATION_EVENT_BASE(AnimationPlaybackEvent, isAnimationPlaybackEvent())
