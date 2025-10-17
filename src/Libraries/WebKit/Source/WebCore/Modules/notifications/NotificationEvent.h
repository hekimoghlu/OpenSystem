/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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

#if ENABLE(NOTIFICATION_EVENT)

#include "ExtendableEvent.h"
#include "ExtendableEventInit.h"
#include "Notification.h"
#include "NotificationEventType.h"

namespace WebCore {

struct NotificationEventInit : ExtendableEventInit {
    RefPtr<Notification> notification;
    String action;
};

class NotificationEvent final : public ExtendableEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(NotificationEvent);
public:
    ~NotificationEvent();

    using Init = NotificationEventInit;

    static Ref<NotificationEvent> create(const AtomString&, Init&&, IsTrusted = IsTrusted::No);
    static Ref<NotificationEvent> create(const AtomString&, Notification*, const String& action, IsTrusted = IsTrusted::No);

    Notification* notification() { return m_notification.get(); }
    const String& action() { return m_action; }

private:
    NotificationEvent(const AtomString&, NotificationEventInit&&, Notification*, const String& action, IsTrusted = IsTrusted::No);

    RefPtr<Notification> m_notification;
    String m_action;
};

} // namespace WebCore

#endif // ENABLE(NOTIFICATION_EVENT)
