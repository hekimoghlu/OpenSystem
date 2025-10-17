/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 9, 2022.
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

#if ENABLE(WIRELESS_PLAYBACK_TARGET_AVAILABILITY_API)

#include "Event.h"

namespace WebCore {

class WebKitPlaybackTargetAvailabilityEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebKitPlaybackTargetAvailabilityEvent);
public:

    static Ref<WebKitPlaybackTargetAvailabilityEvent> create(const AtomString& eventType, bool available)
    {
        return adoptRef(*new WebKitPlaybackTargetAvailabilityEvent(eventType, available));
    }

    struct Init : EventInit {
        String availability;
    };

    static Ref<WebKitPlaybackTargetAvailabilityEvent> create(const AtomString& eventType, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new WebKitPlaybackTargetAvailabilityEvent(eventType, initializer, isTrusted));
    }

    String availability() const { return m_availability; }

private:
    explicit WebKitPlaybackTargetAvailabilityEvent(const AtomString& eventType, bool available);
    WebKitPlaybackTargetAvailabilityEvent(const AtomString& eventType, const Init&, IsTrusted);

    String m_availability;
};

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET_AVAILABILITY_API)
