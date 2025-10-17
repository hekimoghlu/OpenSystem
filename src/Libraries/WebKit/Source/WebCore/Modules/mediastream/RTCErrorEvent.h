/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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

#if ENABLE(WEB_RTC)

#include "Event.h"
#include "RTCError.h"

namespace WebCore {

class RTCErrorEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCErrorEvent);
public:
    struct Init : EventInit {
        RefPtr<RTCError> error;
    };
    static Ref<RTCErrorEvent> create(const AtomString& type, Init&& init, IsTrusted isTrusted = IsTrusted::No) { return adoptRef(*new RTCErrorEvent(type, WTFMove(init), isTrusted)); }
    static Ref<RTCErrorEvent> create(const AtomString& type, RefPtr<RTCError>&& error) { return create(type, Init { { }, WTFMove(error) }, IsTrusted::Yes); }

    RTCError& error() const { return m_error.get(); }

private:
    RTCErrorEvent(const AtomString& type, Init&&, IsTrusted);

    Ref<RTCError> m_error;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
