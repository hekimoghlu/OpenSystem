/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#include "AnimationPlaybackEvent.h"

#include "WebAnimationUtilities.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AnimationPlaybackEvent);

AnimationPlaybackEvent::AnimationPlaybackEvent(const AtomString& type, const AnimationPlaybackEventInit& initializer, IsTrusted isTrusted)
    : AnimationEventBase(EventInterfaceType::AnimationPlaybackEvent, type, initializer, isTrusted)
    , m_timelineTime(initializer.timelineTime)
    , m_currentTime(initializer.currentTime)
{
}

AnimationPlaybackEvent::AnimationPlaybackEvent(const AtomString& type, WebAnimation* animation, std::optional<WebAnimationTime> scheduledTime, std::optional<WebAnimationTime> timelineTime, std::optional<WebAnimationTime> currentTime)
    : AnimationEventBase(EventInterfaceType::AnimationPlaybackEvent, type, animation, scheduledTime)
{
    if (timelineTime)
        m_timelineTime = *timelineTime;
    if (currentTime)
        m_currentTime = *currentTime;
}

AnimationPlaybackEvent::~AnimationPlaybackEvent() = default;

} // namespace WebCore
