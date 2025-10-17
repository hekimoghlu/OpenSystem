/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#include "NotificationEvent.h"

#if ENABLE(NOTIFICATION_EVENT)

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(NotificationEvent);

Ref<NotificationEvent> NotificationEvent::create(const AtomString& type, Init&& init, IsTrusted isTrusted)
{
    auto notification = init.notification;
    ASSERT(notification);
    auto action = init.action;
    return adoptRef(*new NotificationEvent(type, WTFMove(init), notification.get(), action, isTrusted));
}

Ref<NotificationEvent> NotificationEvent::create(const AtomString& type, Notification* notification, const String& action, IsTrusted isTrusted)
{
    return adoptRef(*new NotificationEvent(type, { }, notification, action, isTrusted));
}

NotificationEvent::NotificationEvent(const AtomString& type, NotificationEventInit&& eventInit, Notification* notification, const String& action, IsTrusted isTrusted)
    : ExtendableEvent(EventInterfaceType::NotificationEvent, type, WTFMove(eventInit), isTrusted)
    , m_notification(notification)
    , m_action(action)
{
}

NotificationEvent::~NotificationEvent() = default;

} // namespace WebCore

#endif // ENABLE(NOTIFICATION_EVENT)


