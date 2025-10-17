/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "Event.h"
#include "MediaKeyMessageType.h"
#include <JavaScriptCore/Forward.h>

namespace WebCore {

struct MediaKeyMessageEventInit;

class MediaKeyMessageEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MediaKeyMessageEvent);
public:
    using Type = MediaKeyMessageType;
    using Init = MediaKeyMessageEventInit;

    virtual ~MediaKeyMessageEvent();

    static Ref<MediaKeyMessageEvent> create(const AtomString& type, const MediaKeyMessageEventInit& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new MediaKeyMessageEvent(type, initializer, isTrusted));
    }

    Type messageType() const { return m_messageType; }
    RefPtr<JSC::ArrayBuffer> message() const;

private:
    MediaKeyMessageEvent(const AtomString&, const MediaKeyMessageEventInit&, IsTrusted);

    MediaKeyMessageType m_messageType;
    RefPtr<JSC::ArrayBuffer> m_message;
};

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
