/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "Event.h"
#include "WebKitMediaKeyError.h"

namespace WebCore {

class WebKitMediaKeyMessageEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebKitMediaKeyMessageEvent);
public:
    virtual ~WebKitMediaKeyMessageEvent();

    static Ref<WebKitMediaKeyMessageEvent> create(const AtomString& type, Uint8Array* message, const String& destinationURL)
    {
        return adoptRef(*new WebKitMediaKeyMessageEvent(type, message, destinationURL));
    }

    struct Init : EventInit {
        RefPtr<Uint8Array> message;
        String destinationURL;
    };

    static Ref<WebKitMediaKeyMessageEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new WebKitMediaKeyMessageEvent(type, initializer, isTrusted));
    }

    Uint8Array* message() const { return m_message.get(); }
    String destinationURL() const { return m_destinationURL; }

private:
    WebKitMediaKeyMessageEvent(const AtomString& type, Uint8Array* message, const String& destinationURL);
    WebKitMediaKeyMessageEvent(const AtomString& type, const Init&, IsTrusted);

    RefPtr<Uint8Array> m_message;
    String m_destinationURL;
};

} // namespace WebCore

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
