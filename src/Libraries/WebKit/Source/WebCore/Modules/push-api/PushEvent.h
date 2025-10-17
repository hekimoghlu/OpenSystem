/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#include "ExtendableEvent.h"
#include "Notification.h"
#include "NotificationData.h"
#include "PushEventInit.h"

namespace WebCore {

class PushMessageData;

class PushEvent final : public ExtendableEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PushEvent);
public:
    static Ref<PushEvent> create(const AtomString&, PushEventInit&&, IsTrusted = IsTrusted::No);
    static Ref<PushEvent> create(const AtomString&, ExtendableEventInit&&, std::optional<Vector<uint8_t>>&&, IsTrusted);
    ~PushEvent();

    PushMessageData* data() { return m_data.get(); }

#if ENABLE(DECLARATIVE_WEB_PUSH) && ENABLE(NOTIFICATIONS)
    static Ref<PushEvent> create(const AtomString&, ExtendableEventInit&&, Ref<Notification>, std::optional<uint64_t> appBadge, IsTrusted);

    Notification* notification();
    std::optional<uint64_t> appBadge();

    Notification* proposedNotification() const { return m_proposedNotification.get(); }
    std::optional<uint64_t> proposedAppBadge() const { return m_proposedAppBadge; }

    void setUpdatedNotification(Notification* notification) { m_updatedNotification = notification; }
    std::optional<NotificationData> updatedNotificationData() const;

    void setUpdatedAppBadge(std::optional<uint64_t>&& updatedAppBadge) { m_updatedAppBadge = WTFMove(updatedAppBadge); }
    const std::optional<std::optional<uint64_t>>& updatedAppBadge() const { return m_updatedAppBadge; }
#endif // ENABLE(DECLARATIVE_WEB_PUSH) && ENABLE(NOTIFICATIONS)

private:
    PushEvent(const AtomString&, ExtendableEventInit&&, std::optional<Vector<uint8_t>>&&, IsTrusted);

    RefPtr<PushMessageData> m_data;

#if ENABLE(DECLARATIVE_WEB_PUSH) && ENABLE(NOTIFICATIONS)
    PushEvent(const AtomString&, ExtendableEventInit&&, std::optional<Vector<uint8_t>>&&, RefPtr<Notification>, std::optional<uint64_t> appBadge, IsTrusted);

    RefPtr<Notification> m_proposedNotification;
    std::optional<uint64_t> m_proposedAppBadge;

    RefPtr<Notification> m_updatedNotification;
    std::optional<std::optional<uint64_t>> m_updatedAppBadge;
#endif // ENABLE(DECLARATIVE_WEB_PUSH) && ENABLE(NOTIFICATIONS)
};

} // namespace WebCore
