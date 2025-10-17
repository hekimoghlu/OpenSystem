/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

#include "BackgroundFetchInformation.h"
#include "NotificationClient.h"
#include "NotificationEventType.h"
#include "PushSubscriptionData.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerContextData.h"
#include "ServiceWorkerFetch.h"
#include "ServiceWorkerIdentifier.h"
#include "Settings.h"
#include "Timer.h"
#include "WorkerThread.h"
#include <wtf/OptionSet.h>

namespace WebCore {

class CacheStorageProvider;
class ContentSecurityPolicyResponseHeaders;
class ExtendableEvent;
class MessagePortChannel;
class SerializedScriptValue;
class WorkerObjectProxy;
struct MessageWithMessagePorts;
struct NotificationData;
struct NotificationPayload;

enum class AdvancedPrivacyProtections : uint16_t;

class ServiceWorkerThread : public WorkerThread {
public:
    template<typename... Args> static Ref<ServiceWorkerThread> create(Args&&... args)
    {
        return adoptRef(*new ServiceWorkerThread(std::forward<Args>(args)...));
    }
    virtual ~ServiceWorkerThread();

    WorkerObjectProxy& workerObjectProxy() const { return m_workerObjectProxy; }

    void start(Function<void(const String&, bool)>&&);

    void willPostTaskToFireInstallEvent();
    void willPostTaskToFireActivateEvent();
    void willPostTaskToFireMessageEvent();
    void willPostTaskToFirePushSubscriptionChangeEvent();

    void queueTaskToFireFetchEvent(Ref<ServiceWorkerFetch::Client>&&, ResourceRequest&&, String&& referrer, FetchOptions&&, SWServerConnectionIdentifier, FetchIdentifier, bool isServiceWorkerNavigationPreloadEnabled, String&& clientIdentifier, String&& resultingClientIdentifier);
    void queueTaskToPostMessage(MessageWithMessagePorts&&, ServiceWorkerOrClientData&& sourceData);
    void queueTaskToFireInstallEvent();
    void queueTaskToFireActivateEvent();
    void queueTaskToFirePushEvent(std::optional<Vector<uint8_t>>&&, std::optional<NotificationPayload>&&, Function<void(bool, std::optional<NotificationPayload>&&)>&&);
#if ENABLE(DECLARATIVE_WEB_PUSH)
    void queueTaskToFireDeclarativePushEvent(NotificationPayload&&, Function<void(bool, std::optional<NotificationPayload>&&)>&&);
#endif
    void queueTaskToFirePushSubscriptionChangeEvent(std::optional<PushSubscriptionData>&& newSubscriptionData, std::optional<PushSubscriptionData>&& oldSubscriptionData);
#if ENABLE(NOTIFICATION_EVENT)
    void queueTaskToFireNotificationEvent(NotificationData&&, NotificationEventType, Function<void(bool)>&&);
#endif
    void queueTaskToFireBackgroundFetchEvent(BackgroundFetchInformation&&, Function<void(bool)>&&);
    void queueTaskToFireBackgroundFetchClickEvent(BackgroundFetchInformation&&, Function<void(bool)>&&);

    ServiceWorkerIdentifier identifier() const { return m_serviceWorkerIdentifier; }
    std::optional<ServiceWorkerJobDataIdentifier> jobDataIdentifier() const { return m_jobDataIdentifier; }
    bool doesHandleFetch() const { return m_doesHandleFetch; }

    void startFetchEventMonitoring();
    void stopFetchEventMonitoring() { m_isHandlingFetchEvent = false; }
    void startFunctionalEventMonitoring();
    void stopFunctionalEventMonitoring() { m_isHandlingFunctionalEvent = false; }
    void startNotificationPayloadFunctionalEventMonitoring();
    void stopNotificationPayloadFunctionalEventMonitoring() { m_isHandlingNotificationPayloadFunctionalEvent = false; }

protected:
    Ref<WorkerGlobalScope> createWorkerGlobalScope(const WorkerParameters&, Ref<SecurityOrigin>&&, Ref<SecurityOrigin>&& topOrigin) final;
    void runEventLoop() override;

private:
    WEBCORE_EXPORT ServiceWorkerThread(ServiceWorkerContextData&&, ServiceWorkerData&&, String&& userAgent, WorkerThreadMode, const Settings::Values&, WorkerLoaderProxy&, WorkerDebuggerProxy&, WorkerBadgeProxy&, IDBClient::IDBConnectionProxy*, SocketProvider*, std::unique_ptr<NotificationClient>&&, PAL::SessionID, std::optional<uint64_t>, OptionSet<AdvancedPrivacyProtections>);

    ASCIILiteral threadName() const final { return "WebCore: ServiceWorker"_s; }
    void finishedEvaluatingScript() final;

    void finishedFiringInstallEvent(bool hasRejectedAnyPromise);
    void finishedFiringActivateEvent();
    void finishedFiringMessageEvent();
    void finishedFiringPushSubscriptionChangeEvent();
    void finishedStarting();

    void startHeartBeatTimer();
    void heartBeatTimerFired();
    void installEventTimerFired();

    ServiceWorkerIdentifier m_serviceWorkerIdentifier;
    std::optional<ServiceWorkerJobDataIdentifier> m_jobDataIdentifier;
    std::optional<ServiceWorkerContextData> m_contextData; // Becomes std::nullopt after the ServiceWorkerGlobalScope has been created.
    std::optional<ServiceWorkerData> m_workerData; // Becomes std::nullopt after the ServiceWorkerGlobalScope has been created.
    WorkerObjectProxy& m_workerObjectProxy;
    bool m_doesHandleFetch { false };

    bool m_isHandlingFetchEvent { false };
    bool m_isHandlingFunctionalEvent { false };
    bool m_isHandlingNotificationPayloadFunctionalEvent { false };
    uint64_t m_pushSubscriptionChangeEventCount { 0 };
    uint64_t m_messageEventCount { 0 };
    enum class State { Idle, Starting, Installing, Activating };
    State m_state { State::Idle };
    bool m_ongoingHeartBeatCheck { false };

    static constexpr Seconds heartBeatTimeout { 60_s };
    static constexpr Seconds heartBeatTimeoutForTest { 1_s };
    Seconds m_heartBeatTimeout { heartBeatTimeout };
    Timer m_heartBeatTimer;
    std::unique_ptr<NotificationClient> m_notificationClient;
};

} // namespace WebCore
