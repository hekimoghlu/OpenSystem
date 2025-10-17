/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "EventNames.h"

namespace WebCore {

class CloseEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(CloseEvent);
public:
    static Ref<CloseEvent> create(bool wasClean, unsigned short code, const String& reason)
    {
        return adoptRef(*new CloseEvent(wasClean, code, reason));
    }

    struct Init : EventInit {
        bool wasClean { false };
        unsigned short code { 0 };
        String reason;
    };

    static Ref<CloseEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new CloseEvent(type, initializer, isTrusted));
    }

    bool wasClean() const { return m_wasClean; }
    unsigned short code() const { return m_code; }
    String reason() const { return m_reason; }

private:
    CloseEvent(bool wasClean, int code, const String& reason)
        : Event(EventInterfaceType::CloseEvent, eventNames().closeEvent, CanBubble::No, IsCancelable::No)
        , m_wasClean(wasClean)
        , m_code(code)
        , m_reason(reason)
    {
    }

    CloseEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
        : Event(EventInterfaceType::CloseEvent, type, initializer, isTrusted)
        , m_wasClean(initializer.wasClean)
        , m_code(initializer.code)
        , m_reason(initializer.reason)
    {
    }

    bool m_wasClean;
    unsigned short m_code;
    String m_reason;
};

} // namespace WebCore
