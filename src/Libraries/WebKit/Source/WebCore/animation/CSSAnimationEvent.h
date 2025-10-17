/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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

#include "StyleOriginatedAnimationEvent.h"

namespace WebCore {

class CSSAnimationEvent final : public StyleOriginatedAnimationEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSAnimationEvent);
public:
    static Ref<CSSAnimationEvent> create(const AtomString& type, WebAnimation* animation, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier, const String& animationName)
    {
        return adoptRef(*new CSSAnimationEvent(type, animation, scheduledTime, elapsedTime, pseudoElementIdentifier, animationName));
    }

    struct Init : EventInit {
        String animationName;
        double elapsedTime { 0 };
        String pseudoElement;
    };

    static Ref<CSSAnimationEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new CSSAnimationEvent(type, initializer, isTrusted));
    }

    virtual ~CSSAnimationEvent();

    bool isCSSAnimationEvent() const final { return true; }

    const String& animationName() const { return m_animationName; }

private:
    CSSAnimationEvent(const AtomString& type, WebAnimation*, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>&, const String& animationName);
    CSSAnimationEvent(const AtomString&, const Init&, IsTrusted);

    String m_animationName;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ANIMATION_EVENT_BASE(CSSAnimationEvent, isCSSAnimationEvent())
