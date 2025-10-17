/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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
#include "WebKitPlaybackTargetAvailabilityEvent.h"

#if ENABLE(WIRELESS_PLAYBACK_TARGET_AVAILABILITY_API)

#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebKitPlaybackTargetAvailabilityEvent);

static const AtomString& stringForPlaybackTargetAvailability(bool available)
{
    static MainThreadNeverDestroyed<const AtomString> availableString("available"_s);
    static MainThreadNeverDestroyed<const AtomString> notAvailableString("not-available"_s);

    return available ? availableString : notAvailableString;
}

WebKitPlaybackTargetAvailabilityEvent::WebKitPlaybackTargetAvailabilityEvent(const AtomString& eventType, bool available)
    : Event(EventInterfaceType::WebKitPlaybackTargetAvailabilityEvent, eventType, CanBubble::No, IsCancelable::No)
    , m_availability(stringForPlaybackTargetAvailability(available))
{
}

WebKitPlaybackTargetAvailabilityEvent::WebKitPlaybackTargetAvailabilityEvent(const AtomString& eventType, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::WebKitPlaybackTargetAvailabilityEvent, eventType, initializer, isTrusted)
    , m_availability(initializer.availability)
{
}

} // namespace WebCore

#endif // ENABLE(WIRELESS_PLAYBACK_TARGET_AVAILABILITY_API)
