/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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

#include "PushServiceConnection.h"
#include "WebPushMessage.h"
#include <WebCore/ExceptionData.h>
#include <WebCore/PushDatabase.h>
#include <WebCore/PushSubscriptionData.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Deque.h>
#include <wtf/Expected.h>
#include <wtf/Function.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebPushD {

class GetSubscriptionRequest;
class PushServiceRequest;
class SubscribeRequest;
class UnsubscribeRequest;

class PushService : public RefCountedAndCanMakeWeakPtr<PushService> {
    WTF_MAKE_TZONE_ALLOCATED(PushService);
public:
    friend class SubscribeRequest;
    friend class UnsubscribeRequest;

    using IncomingPushMessageHandler = Function<void(const WebCore::PushSubscriptionSetIdentifier&, WebKit::WebPushMessage&&)>;

    static void create(const String& incomingPushServiceName, const String& databasePath, IncomingPushMessageHandler&&, CompletionHandler<void(RefPtr<PushService>&&)>&&);
    static void createMockService(IncomingPushMessageHandler&&, CompletionHandler<void(RefPtr<PushService>&&)>&&);
    ~PushService();

    PushServiceConnection& connection() { return m_connection; }
    Ref<PushServiceConnection> protectedConnection() { return m_connection; }
    WebCore::PushDatabase& database() { return m_database; }
    Ref<WebCore::PushDatabase> protectedDatabase() { return m_database; }

    Vector<String> enabledTopics() { return m_connection->enabledTopics(); }
    Vector<String> ignoredTopics() { return m_connection->ignoredTopics(); }

    void getSubscription(const WebCore::PushSubscriptionSetIdentifier&, const String& scope, CompletionHandler<void(const Expected<std::optional<WebCore::PushSubscriptionData>, WebCore::ExceptionData>&)>&&);
    void subscribe(const WebCore::PushSubscriptionSetIdentifier&, const String& scope, const Vector<uint8_t>& vapidPublicKey, CompletionHandler<void(const Expected<WebCore::PushSubscriptionData, WebCore::ExceptionData>&)>&&);
    void unsubscribe(const WebCore::PushSubscriptionSetIdentifier&, const String& scope, std::optional<WebCore::PushSubscriptionIdentifier>, CompletionHandler<void(const Expected<bool, WebCore::ExceptionData>&)>&&);

    void incrementSilentPushCount(const WebCore::PushSubscriptionSetIdentifier&, const String& securityOrigin, CompletionHandler<void(unsigned)>&&);

    void setPushesEnabledForSubscriptionSetAndOrigin(const WebCore::PushSubscriptionSetIdentifier&, const String& securityOrigin, bool, CompletionHandler<void()>&&);

    void removeRecordsForSubscriptionSet(const WebCore::PushSubscriptionSetIdentifier&, CompletionHandler<void(unsigned)>&&);
    void removeRecordsForSubscriptionSetAndOrigin(const WebCore::PushSubscriptionSetIdentifier&, const String& securityOrigin, CompletionHandler<void(unsigned)>&&);
    void removeRecordsForBundleIdentifierAndDataStore(const String& bundleIdentifier, const std::optional<WTF::UUID>& dataStoreIdentifier, CompletionHandler<void(unsigned)>&&);

    void didCompleteGetSubscriptionRequest(GetSubscriptionRequest&);
    void didCompleteSubscribeRequest(SubscribeRequest&);
    void didCompleteUnsubscribeRequest(UnsubscribeRequest&);

    void setPublicTokenForTesting(Vector<uint8_t>&&);
    void didReceivePublicToken(Vector<uint8_t>&&);
    void didReceivePushMessage(NSString *topic, NSDictionary *userInfo, CompletionHandler<void()>&& = [] { });

#if PLATFORM(IOS)
    void updateSubscriptionSetState(const String& allowedBundleIdentifier, const HashSet<String>& webClipIdentifiers, CompletionHandler<void()>&&);
#endif

private:
    PushService(Ref<PushServiceConnection>&&, Ref<WebCore::PushDatabase>&&, IncomingPushMessageHandler&&);

    using PushServiceRequestMap = HashMap<String, Deque<Ref<PushServiceRequest>>>;
    void enqueuePushServiceRequest(PushServiceRequestMap&, Ref<PushServiceRequest>&&);
    void finishedPushServiceRequest(PushServiceRequestMap&, PushServiceRequest&);

    void removeRecordsImpl(const WebCore::PushSubscriptionSetIdentifier&, const std::optional<String>& securityOrigin, CompletionHandler<void(unsigned)>&&);

    void updateTopicLists(CompletionHandler<void()>&&);

    Ref<PushServiceConnection> m_connection;
    Ref<WebCore::PushDatabase> m_database;

    IncomingPushMessageHandler m_incomingPushMessageHandler;

    PushServiceRequestMap m_getSubscriptionRequests;
    PushServiceRequestMap m_subscribeRequests;
    PushServiceRequestMap m_unsubscribeRequests;

    size_t m_topicCount { 0 };
};

} // namespace WebPushD
