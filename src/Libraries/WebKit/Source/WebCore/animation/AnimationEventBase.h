/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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

#include "Event.h"
#include <wtf/Markable.h>

namespace WebCore {

class WebAnimation;

class AnimationEventBase : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(AnimationEventBase);
public:
    virtual ~AnimationEventBase();

    virtual bool isAnimationPlaybackEvent() const { return false; }
    virtual bool isCSSAnimationEvent() const { return false; }
    virtual bool isCSSTransitionEvent() const { return false; }

    WebAnimation* animation() const { return m_animation.get(); }
    std::optional<Seconds> scheduledTime() const { return m_scheduledTime; }

protected:
    AnimationEventBase(enum EventInterfaceType, const AtomString&, WebAnimation*, std::optional<Seconds> scheduledTime);
    AnimationEventBase(enum EventInterfaceType, const AtomString&, const EventInit&, IsTrusted);

private:
    RefPtr<WebAnimation> m_animation;
    Markable<Seconds, Seconds::MarkableTraits> m_scheduledTime;
};

}

#define SPECIALIZE_TYPE_TRAITS_ANIMATION_EVENT_BASE(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
static bool isType(const WebCore::AnimationEventBase& value) { return value.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

