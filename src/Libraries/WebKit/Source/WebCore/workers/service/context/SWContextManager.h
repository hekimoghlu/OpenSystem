/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#include "ExceptionOr.h"
#include "PageIdentifier.h"
#include "PushSubscriptionData.h"
#include "ServiceWorkerClientData.h"
#include "ServiceWorkerClientQueryOptions.h"
#include "ServiceWorkerIdentifier.h"
#include "ServiceWorkerThreadProxy.h"
#include <wtf/AbstractRefCounted.h>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URLHash.h>

namespace WebCore {

class SerializedScriptValue;
class ServiceWorkerGlobalScope;

struct NotificationPayload;

class SWContextManager {
public:
    WEBCORE_EXPORT static SWContextManager& singleton();

    class Connection : public AbstractRefCounted {
    public:
        virtual ~Connection() { }

        virtual void establishConnection(CompletionHandler<void()>&&) = 0;
        virtual void postMessageToServiceWorkerClient(const ScriptExecutionContextIdentifier& destinationIdentifier, const MessageWithMessagePorts&, ServiceWorkerIdentifier source, const String& sourceOrigin) = 0;
        virtual void serviceWorkerStarted(std::optional<ServiceWorkerJobDataIdentifier>, ServiceWorkerIdentifier, bool doesHandleFetch) = 0;
        virtual void serviceWorkerFailedToStart(std::optional<ServiceWorkerJobDataIdentifier>, ServiceWorkerIdentifier, const String& message) = 0;
        virtual void didFinishInstall(std::optional<ServiceWorkerJobDataIdentifier>, ServiceWorkerIdentifier, bool wasSuccessful) = 0;
        virtual void didFinishActivation(ServiceWorkerIdentifier) = 0;
        virtual void setServiceWorkerHasPendingEvents(ServiceWorkerIdentifier, bool) = 0;
        virtual void workerTerminated(ServiceWorkerIdentifier) = 0;
        virtual void skipWaiting(ServiceWorkerIdentifier, CompletionHandler<void()>&&) = 0;
        virtual void setScriptResource(ServiceWorkerIdentifier, const URL&, const ServiceWorkerContextData::ImportedScript&) = 0;

        using FindClientByIdentifierCallback = CompletionHandler<void(std::optional<ServiceWorkerClientData>&&)>;
        virtual void findClientByVisibleIdentifier(ServiceWorkerIdentifier, const String&, FindClientByIdentifierCallback&&) = 0;
        virtual void matchAll(ServiceWorkerIdentifier, const ServiceWorkerClientQueryOptions&, ServiceWorkerClientsMatchAllCallback&&) = 0;
        virtual void claim(ServiceWorkerIdentifier, CompletionHandler<void(ExceptionOr<void>&&)>&&) = 0;
        virtual void focus(ScriptExecutionContextIdentifier, CompletionHandler<void(std::optional<ServiceWorkerClientData>&&)>&&) = 0;

        using OpenWindowCallback = CompletionHandler<void(ExceptionOr<std::optional<ServiceWorkerClientData>>&&)>;
        virtual void openWindow(ServiceWorkerIdentifier, const URL&, OpenWindowCallback&&) = 0;

        using NavigateCallback = CompletionHandler<void(ExceptionOr<std::optional<WebCore::ServiceWorkerClientData>>&&)>;
        virtual void navigate(ScriptExecutionContextIdentifier, ServiceWorkerIdentifier, const URL&, NavigateCallback&&) = 0;

        virtual void didFailHeartBeatCheck(ServiceWorkerIdentifier) = 0;
        virtual void setAsInspected(ServiceWorkerIdentifier, bool) = 0;

        virtual bool isThrottleable() const = 0;
        virtual PageIdentifier pageIdentifier() const = 0;

        virtual void stop() = 0;

        virtual void reportConsoleMessage(ServiceWorkerIdentifier, MessageSource, MessageLevel, const String& message, unsigned long requestIdentifier) = 0;

        bool isClosed() const { return m_isClosed; }

        virtual void removeNavigationFetch(SWServerConnectionIdentifier, FetchIdentifier) = 0;

    protected:
        void setAsClosed() { m_isClosed = true; }

    private:
        bool m_isClosed { false };
    };

    WEBCORE_EXPORT void setConnection(Ref<Connection>&&);
    WEBCORE_EXPORT Connection* connection() const;

    WEBCORE_EXPORT void registerServiceWorkerThreadForInstall(Ref<ServiceWorkerThreadProxy>&&);
    WEBCORE_EXPORT ServiceWorkerThreadProxy* serviceWorkerThreadProxy(ServiceWorkerIdentifier) const;
    WEBCORE_EXPORT RefPtr<ServiceWorkerThreadProxy> serviceWorkerThreadProxyFromBackgroundThread(ServiceWorkerIdentifier) const;
    WEBCORE_EXPORT void fireInstallEvent(ServiceWorkerIdentifier);
    WEBCORE_EXPORT void fireActivateEvent(ServiceWorkerIdentifier);
    WEBCORE_EXPORT void firePushEvent(ServiceWorkerIdentifier, std::optional<Vector<uint8_t>>&&, std::optional<NotificationPayload>&&, CompletionHandler<void(bool, std::optional<NotificationPayload>&&)>&&);
    WEBCORE_EXPORT void firePushSubscriptionChangeEvent(ServiceWorkerIdentifier, std::optional<PushSubscriptionData>&& newSubscriptionData, std::optional<PushSubscriptionData>&& oldSubscriptionData);
    WEBCORE_EXPORT void fireNotificationEvent(ServiceWorkerIdentifier, NotificationData&&, NotificationEventType, CompletionHandler<void(bool)>&&);
    WEBCORE_EXPORT void fireBackgroundFetchEvent(ServiceWorkerIdentifier, BackgroundFetchInformation&&, CompletionHandler<void(bool)>&&);
    WEBCORE_EXPORT void fireBackgroundFetchClickEvent(ServiceWorkerIdentifier, BackgroundFetchInformation&&, CompletionHandler<void(bool)>&&);

    WEBCORE_EXPORT void terminateWorker(ServiceWorkerIdentifier, Seconds timeout, Function<void()>&&);

    void forEachServiceWorker(const Function<Function<void(ScriptExecutionContext&)>()>&);

    WEBCORE_EXPORT bool postTaskToServiceWorker(ServiceWorkerIdentifier, Function<void(ServiceWorkerGlobalScope&)>&&);

    using ServiceWorkerCreationCallback = void(uint64_t);
    void setServiceWorkerCreationCallback(ServiceWorkerCreationCallback* callback) { m_serviceWorkerCreationCallback = callback; }

    WEBCORE_EXPORT void stopAllServiceWorkers();

    static constexpr Seconds workerTerminationTimeout { 10_s };
    static constexpr Seconds syncWorkerTerminationTimeout { 100_ms }; // Only used by layout tests.

    WEBCORE_EXPORT void setAsInspected(ServiceWorkerIdentifier, bool);
    WEBCORE_EXPORT void setInspectable(bool);

    WEBCORE_EXPORT void updateRegistrationState(ServiceWorkerRegistrationIdentifier, ServiceWorkerRegistrationState, const std::optional<ServiceWorkerData>&);
    WEBCORE_EXPORT void updateWorkerState(ServiceWorkerIdentifier, ServiceWorkerState);
    WEBCORE_EXPORT void fireUpdateFoundEvent(ServiceWorkerRegistrationIdentifier);
    WEBCORE_EXPORT void setRegistrationLastUpdateTime(ServiceWorkerRegistrationIdentifier, WallTime);
    WEBCORE_EXPORT void setRegistrationUpdateViaCache(ServiceWorkerRegistrationIdentifier, ServiceWorkerUpdateViaCache);
    WEBCORE_EXPORT void removeFetch(ServiceWorkerIdentifier, SWServerConnectionIdentifier, FetchIdentifier, bool isNavigationFetch);

private:
    SWContextManager() = default;

    void startedServiceWorker(std::optional<ServiceWorkerJobDataIdentifier>, ServiceWorkerIdentifier, const String& exceptionMessage, bool doesHandleFetch);
    NO_RETURN_DUE_TO_CRASH void serviceWorkerFailedToTerminate(ServiceWorkerIdentifier);

    void stopWorker(ServiceWorkerThreadProxy&, Seconds, Function<void()>&&);

    HashMap<ServiceWorkerIdentifier, Ref<ServiceWorkerThreadProxy>> m_workerMap WTF_GUARDED_BY_LOCK(m_workerMapLock);
    mutable Lock m_workerMapLock;
    RefPtr<Connection> m_connection;
    ServiceWorkerCreationCallback* m_serviceWorkerCreationCallback { nullptr };

    class ServiceWorkerTerminationRequest {
        WTF_MAKE_TZONE_ALLOCATED(ServiceWorkerTerminationRequest);
    public:
        ServiceWorkerTerminationRequest(SWContextManager&, ServiceWorkerIdentifier, Seconds timeout);

    private:
        Timer m_timeoutTimer;
    };
    HashMap<ServiceWorkerIdentifier, std::unique_ptr<ServiceWorkerTerminationRequest>> m_pendingServiceWorkerTerminationRequests;
};

} // namespace WebCore
