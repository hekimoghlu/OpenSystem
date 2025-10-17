/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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

class CSSTransitionEvent final : public StyleOriginatedAnimationEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CSSTransitionEvent);
public:
    static Ref<CSSTransitionEvent> create(const AtomString& type, WebAnimation* animation, std::optional<Seconds> scheduledTime,  double elapsedTime, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier, const String propertyName)
    {
        return adoptRef(*new CSSTransitionEvent(type, animation, scheduledTime, elapsedTime, pseudoElementIdentifier, propertyName));
    }

    struct Init : EventInit {
        String propertyName;
        double elapsedTime { 0 };
        String pseudoElement;
    };

    static Ref<CSSTransitionEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new CSSTransitionEvent(type, initializer, isTrusted));
    }

    virtual ~CSSTransitionEvent();

    bool isCSSTransitionEvent() const final { return true; }

    const String& propertyName() const { return m_propertyName; }

private:
    CSSTransitionEvent(const AtomString& type, WebAnimation*, std::optional<Seconds> scheduledTime, double elapsedTime, const std::optional<Style::PseudoElementIdentifier>&, const String propertyName);
    CSSTransitionEvent(const AtomString& type, const Init& initializer, IsTrusted);

    String m_propertyName;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ANIMATION_EVENT_BASE(CSSTransitionEvent, isCSSTransitionEvent())
