/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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

#include "CookieStore.h"
#include "FetchIdentifier.h"
#include "NotificationClient.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerContextData.h"
#include "ServiceWorkerFetch.h"
#include "ServiceWorkerRegistration.h"
#include "WorkerGlobalScope.h"
#include <wtf/MonotonicTime.h>
#include <wtf/URLHash.h>

namespace WebCore {

class DeferredPromise;
class ExtendableEvent;
class FetchEvent;
class Page;
class PushEvent;
class ServiceWorkerClient;
class ServiceWorkerClients;
class ServiceWorkerThread;
class WorkerClient;

#if ENABLE(DECLARATIVE_WEB_PUSH)
class DeclarativePushEvent;
#endif

enum class NotificationEventType : bool;

struct ServiceWorkerClientData;

class ServiceWorkerGlobalScope final : public WorkerGlobalScope {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ServiceWorkerGlobalScope);
public:
    static Ref<ServiceWorkerGlobalScope> create(ServiceWorkerContextData&&, ServiceWorkerData&&, const WorkerParameters&, Ref<SecurityOrigin>&&, ServiceWorkerThread&, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy*, SocketProvider*, std::unique_ptr<NotificationClient>&&, std::unique_ptr<WorkerClient>&&);

    ~ServiceWorkerGlobalScope();

    bool isServiceWorkerGlobalScope() const final { return true; }

    ServiceWorkerClients& clients() { return m_clients.get(); }
    ServiceWorkerRegistration& registration() { return m_registration.get(); }
    ServiceWorker& serviceWorker() { return m_serviceWorker.get(); }
    
    void skipWaiting(Ref<DeferredPromise>&&);

    enum EventTargetInterfaceType eventTargetInterface() const final;

    ServiceWorkerThread& thread();

    void updateExtendedEventsSet(ExtendableEvent* newEvent = nullptr);

    const ServiceWorkerContextData::ImportedScript* scriptResource(const URL&) const;
    void setScriptResource(const URL&, ServiceWorkerContextData::ImportedScript&&);

    void didSaveScriptsToDisk(ScriptBuffer&&, HashMap<URL, ScriptBuffer>&& importedScripts);

    const ServiceWorkerContextData& contextData() const { return m_contextData; }
    const CertificateInfo& certificateInfo() const { return m_contextData.certificateInfo; }

    FetchOptions::Destination destination() const final { return FetchOptions::Destination::Serviceworker; }

    WEBCORE_EXPORT Page* serviceWorkerPage();

    void dispatchPushEvent(PushEvent&);
    PushEvent* pushEvent() { return m_pushEvent.get(); }

#if ENABLE(DECLARATIVE_WEB_PUSH)
    void dispatchDeclarativePushEvent(PushEvent&);
    PushEvent* declarativePushEvent() { return m_declarativePushEvent.get(); }
    void clearDeclarativePushEvent();
#endif

    bool hasPendingSilentPushEvent() const { return m_hasPendingSilentPushEvent; }
    void setHasPendingSilentPushEvent(bool value) { m_hasPendingSilentPushEvent = value; }

    constexpr static Seconds userGestureLifetime  { 2_s };
    bool isProcessingUserGesture() const { return m_isProcessingUserGesture; }
    void recordUserGesture();
    void setIsProcessingUserGestureForTesting(bool value) { m_isProcessingUserGesture = value; }

    bool didFirePushEventRecently() const;

    WEBCORE_EXPORT void addConsoleMessage(MessageSource, MessageLevel, const String& message, unsigned long requestIdentifier) final;
    void enableConsoleMessageReporting() { m_consoleMessageReportingEnabled = true; }

    CookieStore& cookieStore();

    using FetchKey = std::pair<SWServerConnectionIdentifier, FetchIdentifier>;
    void addFetchTask(FetchKey, Ref<ServiceWorkerFetch::Client>&&);
    void addFetchEvent(FetchKey, FetchEvent&);
    RefPtr<ServiceWorkerFetch::Client> fetchTask(FetchKey);
    bool hasFetchTask() const;
    void removeFetchTask(FetchKey);
    RefPtr<ServiceWorkerFetch::Client> takeFetchTask(FetchKey);
    void navigationPreloadFailed(FetchKey, ResourceError&&);
    void navigationPreloadIsReady(FetchKey, ResourceResponse&&);

private:
    ServiceWorkerGlobalScope(ServiceWorkerContextData&&, ServiceWorkerData&&, const WorkerParameters&, Ref<SecurityOrigin>&&, ServiceWorkerThread&, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy*, SocketProvider*, std::unique_ptr<NotificationClient>&&, std::unique_ptr<WorkerClient>&&);
    void notifyServiceWorkerPageOfCreationIfNecessary();

    void prepareForDestruction() final;

    Type type() const final { return Type::ServiceWorker; }
    bool hasPendingEvents() const { return !m_extendedEvents.isEmpty(); }

    NotificationClient* notificationClient() final { return m_notificationClient.get(); }

    void resetUserGesture() { m_isProcessingUserGesture = false; }

    ServiceWorkerContextData m_contextData;
    Ref<ServiceWorkerRegistration> m_registration;
    Ref<ServiceWorker> m_serviceWorker;
    Ref<ServiceWorkerClients> m_clients;
    Vector<Ref<ExtendableEvent>> m_extendedEvents;

    uint64_t m_lastRequestIdentifier { 0 };
    HashMap<uint64_t, RefPtr<DeferredPromise>> m_pendingSkipWaitingPromises;
    std::unique_ptr<NotificationClient> m_notificationClient;
    bool m_hasPendingSilentPushEvent { false };
    bool m_isProcessingUserGesture { false };
    Timer m_userGestureTimer;
    RefPtr<PushEvent> m_pushEvent;
#if ENABLE(DECLARATIVE_WEB_PUSH)
    RefPtr<PushEvent> m_declarativePushEvent;
#endif
    MonotonicTime m_lastPushEventTime;
    bool m_consoleMessageReportingEnabled { false };
    RefPtr<CookieStore> m_cookieStore;

    struct FetchTask {
        RefPtr<ServiceWorkerFetch::Client> client;
        std::variant<std::nullptr_t, Ref<FetchEvent>, UniqueRef<ResourceError>, UniqueRef<ResourceResponse>> navigationPreload;
    };
    HashMap<FetchKey, FetchTask> m_ongoingFetchTasks;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ServiceWorkerGlobalScope)
    static bool isType(const WebCore::ScriptExecutionContext& context)
    {
        auto* global = dynamicDowncast<WebCore::WorkerGlobalScope>(context);
        return global && global->type() == WebCore::WorkerGlobalScope::Type::ServiceWorker;
    }
    static bool isType(const WebCore::WorkerGlobalScope& context) { return context.type() == WebCore::WorkerGlobalScope::Type::ServiceWorker; }
SPECIALIZE_TYPE_TRAITS_END()
