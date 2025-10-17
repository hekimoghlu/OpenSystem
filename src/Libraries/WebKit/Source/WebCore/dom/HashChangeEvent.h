/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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

class HashChangeEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(HashChangeEvent);
public:
    static Ref<HashChangeEvent> create(const String& oldURL, const String& newURL)
    {
        return adoptRef(*new HashChangeEvent(oldURL, newURL));
    }

    static Ref<HashChangeEvent> createForBindings()
    {
        return adoptRef(*new HashChangeEvent);
    }

    struct Init : EventInit {
        String oldURL;
        String newURL;
    };

    static Ref<HashChangeEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new HashChangeEvent(type, initializer, isTrusted));
    }

    const String& oldURL() const { return m_oldURL; }
    const String& newURL() const { return m_newURL; }

private:
    HashChangeEvent()
        : Event(EventInterfaceType::HashChangeEvent)
    {
    }

    HashChangeEvent(const String& oldURL, const String& newURL)
        : Event(EventInterfaceType::HashChangeEvent, eventNames().hashchangeEvent, CanBubble::No, IsCancelable::No)
        , m_oldURL(oldURL)
        , m_newURL(newURL)
    {
    }

    HashChangeEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
        : Event(EventInterfaceType::HashChangeEvent, type, initializer, isTrusted)
        , m_oldURL(initializer.oldURL)
        , m_newURL(initializer.newURL)
    {
    }

    String m_oldURL;
    String m_newURL;
};

} // namespace WebCore
