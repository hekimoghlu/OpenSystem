/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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
#include "PushEvent.h"

#include "PushMessageData.h"
#include <JavaScriptCore/JSArrayBuffer.h>
#include <JavaScriptCore/JSArrayBufferView.h>
#include <JavaScriptCore/JSCInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PushEvent);

static Vector<uint8_t> dataFromPushMessageDataInit(PushMessageDataInit& data)
{
    return WTF::switchOn(data, [](RefPtr<JSC::ArrayBuffer>& value) -> Vector<uint8_t> {
        if (!value)
            return { };
        return value->span();
    }, [](RefPtr<JSC::ArrayBufferView>& value) -> Vector<uint8_t> {
        if (!value)
            return { };
        return value->span();
    }, [](String& value) -> Vector<uint8_t> {
        return value.utf8().span();
    });
}

Ref<PushEvent> PushEvent::create(const AtomString& type, PushEventInit&& initializer, IsTrusted isTrusted)
{
    std::optional<Vector<uint8_t>> data;
    if (initializer.data)
        data = dataFromPushMessageDataInit(*initializer.data);
    return adoptRef(*new PushEvent(type, WTFMove(initializer), WTFMove(data), isTrusted));
}

Ref<PushEvent> PushEvent::create(const AtomString& type, ExtendableEventInit&& initializer, std::optional<Vector<uint8_t>>&& data, IsTrusted isTrusted)
{
    return adoptRef(*new PushEvent(type, WTFMove(initializer), WTFMove(data), isTrusted));
}

static inline RefPtr<PushMessageData> pushMessageDataFromOptionalVector(std::optional<Vector<uint8_t>>&& data)
{
    if (!data)
        return nullptr;
    return PushMessageData::create(WTFMove(*data));
}


PushEvent::~PushEvent() = default;

PushEvent::PushEvent(const AtomString& type, ExtendableEventInit&& eventInit, std::optional<Vector<uint8_t>>&& data, IsTrusted isTrusted)
#if ENABLE(DECLARATIVE_WEB_PUSH) && ENABLE(NOTIFICATIONS)
    : PushEvent(type, WTFMove(eventInit), WTFMove(data), nullptr, std::nullopt, isTrusted)
#else
    : ExtendableEvent(EventInterfaceType::PushEvent, type, WTFMove(eventInit), isTrusted)
    , m_data(pushMessageDataFromOptionalVector(WTFMove(data)))
#endif
{
}

#if ENABLE(DECLARATIVE_WEB_PUSH) && ENABLE(NOTIFICATIONS)

Ref<PushEvent> PushEvent::create(const AtomString& type, ExtendableEventInit&& initializer, Ref<Notification> proposedNotification, std::optional<uint64_t> proposedAppBadge, IsTrusted isTrusted)
{
    return adoptRef(*new PushEvent(type, WTFMove(initializer), std::nullopt, WTFMove(proposedNotification), proposedAppBadge, isTrusted));
}

PushEvent::PushEvent(const AtomString& type, ExtendableEventInit&& eventInit, std::optional<Vector<uint8_t>>&& data, RefPtr<Notification> proposedNotification, std::optional<uint64_t> proposedAppBadge, IsTrusted isTrusted)
    : ExtendableEvent(EventInterfaceType::PushEvent, type, WTFMove(eventInit), isTrusted)
    , m_data(pushMessageDataFromOptionalVector(WTFMove(data)))
    , m_proposedNotification(proposedNotification)
    , m_proposedAppBadge(proposedAppBadge)
{
}

Notification* PushEvent::notification()
{
    if (m_updatedNotification)
        return m_updatedNotification.get();

    return m_proposedNotification.get();
}

std::optional<uint64_t> PushEvent::appBadge()
{
    if (m_updatedAppBadge.has_value())
        return m_updatedAppBadge.value();

    return m_proposedAppBadge;
}

std::optional<NotificationData> PushEvent::updatedNotificationData() const
{
    if (RefPtr updatedNotification = m_updatedNotification)
        return updatedNotification->data();

    return std::nullopt;
}

#endif // ENABLE(DECLARATIVE_WEB_PUSH) && ENABLE(NOTIFICATIONS)

} // namespace WebCore
