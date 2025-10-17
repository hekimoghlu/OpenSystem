/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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

#if ENABLE(WEB_PUSH_NOTIFICATIONS)

#include "Connection.h"
#include "MessageReceiver.h"
#include "PushMessageForTesting.h"
#include "WebPushMessage.h"
#include <WebCore/ExceptionData.h>
#include <WebCore/NotificationResources.h>
#include <WebCore/PushPermissionState.h>
#include <WebCore/PushSubscriptionData.h>
#include <WebCore/PushSubscriptionIdentifier.h>
#include <WebCore/SecurityOriginData.h>
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/Identified.h>
#include <wtf/OSObjectPtr.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UUID.h>
#include <wtf/WeakPtr.h>
#include <wtf/spi/darwin/XPCSPI.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS UIWebClip;

namespace WebCore {
class SecurityOriginData;
}

namespace WebKit {
namespace WebPushD {
enum class DaemonMessageType : uint8_t;
struct WebPushDaemonConnectionConfiguration;
}
}

using WebKit::WebPushD::DaemonMessageType;
using WebKit::WebPushD::PushMessageForTesting;
using WebKit::WebPushD::WebPushDaemonConnectionConfiguration;

namespace WebPushD {

enum class PushClientConnectionIdentifierType { };
using PushClientConnectionIdentifier = AtomicObjectIdentifier<PushClientConnectionIdentifierType>;

class PushClientConnection : public RefCounted<PushClientConnection>, public Identified<PushClientConnectionIdentifier>, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(PushClientConnection);
public:
    static RefPtr<PushClientConnection> create(xpc_connection_t, IPC::Decoder&);

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    std::optional<WebCore::PushSubscriptionSetIdentifier> subscriptionSetIdentifierForOrigin(const WebCore::SecurityOriginData&) const;
    const String& hostAppCodeSigningIdentifier() const { return m_hostAppCodeSigningIdentifier; }
    bool hostAppHasPushInjectEntitlement() const { return m_hostAppHasPushInjectEntitlement; };
    std::optional<WTF::UUID> dataStoreIdentifier() const { return m_dataStoreIdentifier; }
    bool declarativeWebPushEnabled() const { return m_declarativeWebPushEnabled; }

    // You almost certainly do not want to use this and should probably use subscriptionSetIdentifierForOrigin instead.
    const String& pushPartitionIfExists() const { return m_pushPartitionString; }

    String debugDescription() const;

    void connectionClosed();

    void didReceiveMessageWithReplyHandler(IPC::Decoder&, Function<void(UniqueRef<IPC::Encoder>&&)>&&) override;

private:
    PushClientConnection(xpc_connection_t, String&& hostAppCodeSigningIdentifier, bool hostAppHasPushInjectEntitlement, String&& pushPartitionString, std::optional<WTF::UUID>&& dataStoreIdentifier, bool declarativeWebPushEnabled);

    // PushClientConnectionMessages
    void setPushAndNotificationsEnabledForOrigin(const String& originString, bool, CompletionHandler<void()>&& replySender);
    void injectPushMessageForTesting(PushMessageForTesting&&, CompletionHandler<void(const String&)>&&);
    void injectEncryptedPushMessageForTesting(const String&, CompletionHandler<void(bool)>&&);
    void getPendingPushMessage(CompletionHandler<void(const std::optional<WebKit::WebPushMessage>&)>&& replySender);
    void getPendingPushMessages(CompletionHandler<void(const Vector<WebKit::WebPushMessage>&)>&& replySender);
    void subscribeToPushService(URL&& scopeURL, const Vector<uint8_t>& applicationServerKey, CompletionHandler<void(const Expected<WebCore::PushSubscriptionData, WebCore::ExceptionData>&)>&& replySender);
    void unsubscribeFromPushService(URL&& scopeURL, std::optional<WebCore::PushSubscriptionIdentifier>, CompletionHandler<void(const Expected<bool, WebCore::ExceptionData>&)>&& replySender);
    void getPushSubscription(URL&& scopeURL, CompletionHandler<void(const Expected<std::optional<WebCore::PushSubscriptionData>, WebCore::ExceptionData>&)>&& replySender);
    void incrementSilentPushCount(WebCore::SecurityOriginData&&, CompletionHandler<void(unsigned)>&&);
    void removeAllPushSubscriptions(CompletionHandler<void(unsigned)>&&);
    void removePushSubscriptionsForOrigin(WebCore::SecurityOriginData&&, CompletionHandler<void(unsigned)>&&);
    void setPublicTokenForTesting(const String& publicToken, CompletionHandler<void()>&&);
    void initializeConnection(WebPushDaemonConnectionConfiguration&&);
    void getPushPermissionState(WebCore::SecurityOriginData&&, CompletionHandler<void(WebCore::PushPermissionState)>&&);
    void requestPushPermission(WebCore::SecurityOriginData&&, CompletionHandler<void(bool)>&&);
    void setHostAppAuditTokenData(const Vector<uint8_t>&);
    void getPushTopicsForTesting(CompletionHandler<void(Vector<String>, Vector<String>)>&&);

    void showNotification(const WebCore::NotificationData&, RefPtr<WebCore::NotificationResources>, CompletionHandler<void()>&&);
    void getNotifications(const URL& registrationURL, const String& tag, CompletionHandler<void(Expected<Vector<WebCore::NotificationData>, WebCore::ExceptionData>&&)>&&);
    void cancelNotification(WebCore::SecurityOriginData&&, const WTF::UUID& notificationID);
    void setAppBadge(WebCore::SecurityOriginData&&, std::optional<uint64_t>);
    void getAppBadgeForTesting(CompletionHandler<void(std::optional<uint64_t>)>&&);
    void setProtocolVersionForTesting(unsigned, CompletionHandler<void()>&&);

    OSObjectPtr<xpc_connection_t> m_xpcConnection;
    String m_hostAppCodeSigningIdentifier;
    bool m_hostAppHasPushInjectEntitlement { false };
    String m_pushPartitionString;
    Markable<WTF::UUID> m_dataStoreIdentifier;
    bool m_declarativeWebPushEnabled { false };
};

} // namespace WebPushD

#endif // ENABLE(WEB_PUSH_NOTIFICATIONS)
