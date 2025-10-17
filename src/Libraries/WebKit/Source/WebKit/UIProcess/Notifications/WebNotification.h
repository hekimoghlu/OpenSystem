/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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

#include "APIDictionary.h"
#include "APISecurityOrigin.h"
#include "Connection.h"
#include "WebNotificationIdentifier.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/NotificationData.h>
#include <wtf/Identified.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
enum class NotificationDirection : uint8_t;
struct NotificationData;
}

namespace WebKit {

class WebNotification : public API::ObjectImpl<API::Object::Type::Notification>, public Identified<WebNotificationIdentifier> {
public:
    static Ref<WebNotification> createNonPersistent(const WebCore::NotificationData& data, std::optional<WebPageProxyIdentifier> pageIdentifier, IPC::Connection& sourceConnection)
    {
        ASSERT(!data.isPersistent());
        return adoptRef(*new WebNotification(data, pageIdentifier, std::nullopt, &sourceConnection));
    }

    static Ref<WebNotification> createPersistent(const WebCore::NotificationData& data, const std::optional<WTF::UUID>& dataStoreIdentifier, IPC::Connection* sourceConnection)
    {
        ASSERT(data.isPersistent());
        return adoptRef(*new WebNotification(data, std::nullopt, dataStoreIdentifier, sourceConnection));
    }

    const String& title() const { return m_data.title; }
    const String& body() const { return m_data.body; }
    const String& iconURL() const { return m_data.iconURL; }
    const String& tag() const { return m_data.tag; }
    const String& lang() const { return m_data.language; }
    WebCore::NotificationDirection dir() const { return m_data.direction; }
    const WTF::UUID& coreNotificationID() const { return m_data.notificationID; }
    const std::optional<WTF::UUID>& dataStoreIdentifier() const { return m_dataStoreIdentifier; }
    PAL::SessionID sessionID() const { return m_data.sourceSession; }

    const WebCore::NotificationData& data() const { return m_data; }
    bool isPersistentNotification() const { return !m_data.serviceWorkerRegistrationURL.isEmpty(); }

    const API::SecurityOrigin* origin() const { return m_origin.get(); }
    API::SecurityOrigin* origin() { return m_origin.get(); }

    std::optional<WebPageProxyIdentifier> pageIdentifier() const { return m_pageIdentifier; }
    RefPtr<IPC::Connection> sourceConnection() const { return m_sourceConnection.get(); }

private:
    WebNotification(const WebCore::NotificationData&, std::optional<WebPageProxyIdentifier>, const std::optional<WTF::UUID>& dataStoreIdentifier, IPC::Connection*);

    WebCore::NotificationData m_data;
    RefPtr<API::SecurityOrigin> m_origin;
    Markable<WebPageProxyIdentifier> m_pageIdentifier;
    std::optional<WTF::UUID> m_dataStoreIdentifier;
    ThreadSafeWeakPtr<IPC::Connection> m_sourceConnection;
};

inline bool isNotificationIDValid(uint64_t id)
{
    // This check makes sure that the ID is not equal to values needed by
    // HashMap for bucketing.
    return id && id != static_cast<uint64_t>(-1);
}

} // namespace WebKit
