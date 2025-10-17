/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

#include "ActiveDOMObject.h"
#include "AddEventListenerOptions.h"
#include "EventTarget.h"
#include "IDLTypes.h"
#include "JSDOMPromiseDeferredForward.h"
#include "MessageEvent.h"
#include "PushPermissionState.h"
#include "PushSubscription.h"
#include "SWClientConnection.h"
#include "SWServer.h"
#include "ServiceWorkerJobClient.h"
#include "ServiceWorkerRegistration.h"
#include "ServiceWorkerRegistrationOptions.h"
#include "WorkerType.h"
#include <wtf/Forward.h>
#include <wtf/Threading.h>

namespace WebCore {

class DeferredPromise;
class NavigatorBase;
class ServiceWorker;
class TrustedScriptURL;

enum class ServiceWorkerUpdateViaCache : uint8_t;
enum class WorkerType : bool;

struct CookieChangeSubscription;

class ServiceWorkerContainer final : public EventTarget, public ActiveDOMObject, public ServiceWorkerJobClient {
    WTF_MAKE_NONCOPYABLE(ServiceWorkerContainer);
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ServiceWorkerContainer);
public:
    static UniqueRef<ServiceWorkerContainer> create(ScriptExecutionContext*, NavigatorBase&);

    ~ServiceWorkerContainer();

    // ActiveDOMObject.
    void ref() const final;
    void deref() const final;

    ServiceWorker* controller() const;

    using ReadyPromise = DOMPromiseProxy<IDLInterface<ServiceWorkerRegistration>>;
    ReadyPromise& ready();

    using RegistrationOptions = ServiceWorkerRegistrationOptions;
    void addRegistration(std::variant<RefPtr<TrustedScriptURL>, String>&&, const RegistrationOptions&, Ref<DeferredPromise>&&);
    void unregisterRegistration(ServiceWorkerRegistrationIdentifier, DOMPromiseDeferred<IDLBoolean>&&);
    void updateRegistration(const URL& scopeURL, const URL& scriptURL, WorkerType, RefPtr<DeferredPromise>&&);

    void getRegistration(const String& clientURL, Ref<DeferredPromise>&&);
    void updateRegistrationState(ServiceWorkerRegistrationIdentifier, ServiceWorkerRegistrationState, const std::optional<ServiceWorkerData>&);
    void updateWorkerState(ServiceWorkerIdentifier, ServiceWorkerState);
    void queueTaskToFireUpdateFoundEvent(ServiceWorkerRegistrationIdentifier);
    void queueTaskToDispatchControllerChangeEvent();

    void postMessage(MessageWithMessagePorts&&, ServiceWorkerData&& sourceData, String&& sourceOrigin);

    void getRegistrations(Ref<DeferredPromise>&&);

    ServiceWorkerRegistration* registration(ServiceWorkerRegistrationIdentifier identifier) const { return m_registrations.get(identifier); }

    void addRegistration(ServiceWorkerRegistration&);
    void removeRegistration(ServiceWorkerRegistration&);

    void subscribeToPushService(ServiceWorkerRegistration&, const Vector<uint8_t>& applicationServerKey, DOMPromiseDeferred<IDLInterface<PushSubscription>>&&);
    void unsubscribeFromPushService(ServiceWorkerRegistrationIdentifier, PushSubscriptionIdentifier, DOMPromiseDeferred<IDLBoolean>&&);
    void getPushSubscription(ServiceWorkerRegistration&, DOMPromiseDeferred<IDLNullable<IDLInterface<PushSubscription>>>&&);
    void getPushPermissionState(ServiceWorkerRegistrationIdentifier, DOMPromiseDeferred<IDLEnumeration<PushPermissionState>>&&);
#if ENABLE(NOTIFICATIONS)
    void getNotifications(const URL&, const String&, DOMPromiseDeferred<IDLSequence<IDLInterface<Notification>>>&&);
#endif

    ServiceWorkerJob* job(ServiceWorkerJobIdentifier);

    void startMessages();

    bool isStopped() const { return m_isStopped; };

    NavigatorBase* navigator() { return &m_navigator; }

    using VoidPromise = DOMPromiseDeferred<void>;
    using NavigationPreloadStatePromise = DOMPromiseDeferred<IDLDictionary<NavigationPreloadState>>;
    void enableNavigationPreload(ServiceWorkerRegistrationIdentifier, VoidPromise&&);
    void disableNavigationPreload(ServiceWorkerRegistrationIdentifier, VoidPromise&&);
    void setNavigationPreloadHeaderValue(ServiceWorkerRegistrationIdentifier, String&&, VoidPromise&&);
    void getNavigationPreloadState(ServiceWorkerRegistrationIdentifier, NavigationPreloadStatePromise&&);

    void addCookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, Vector<CookieChangeSubscription>&&, Ref<DeferredPromise>&&);
    void removeCookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, Vector<CookieChangeSubscription>&&, Ref<DeferredPromise>&&);
    void cookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, Ref<DeferredPromise>&&);

private:
    ServiceWorkerContainer(ScriptExecutionContext*, NavigatorBase&);

    bool addEventListener(const AtomString& eventType, Ref<EventListener>&&, const AddEventListenerOptions& = { }) final;

    void scheduleJob(std::unique_ptr<ServiceWorkerJob>&&);

    void jobFailedWithException(ServiceWorkerJob&, const Exception&) final;
    void jobResolvedWithRegistration(ServiceWorkerJob&, ServiceWorkerRegistrationData&&, ShouldNotifyWhenResolved) final;
    void jobResolvedWithUnregistrationResult(ServiceWorkerJob&, bool unregistrationResult) final;
    void startScriptFetchForJob(ServiceWorkerJob&, FetchOptions::Cache) final;
    void jobFinishedLoadingScript(ServiceWorkerJob&, WorkerFetchResult&&) final;
    void jobFailedLoadingScript(ServiceWorkerJob&, const ResourceError&, Exception&&) final;

    void notifyFailedFetchingScript(ServiceWorkerJob&, const ResourceError&);
    void destroyJob(ServiceWorkerJob&);
    void willSettleRegistrationPromise(bool success);

    ServiceWorkerOrClientIdentifier contextIdentifier() final;
    ScriptExecutionContext* context() final { return scriptExecutionContext(); }

    SWClientConnection& ensureSWClientConnection();
    Ref<SWClientConnection> ensureProtectedSWClientConnection();

    ScriptExecutionContext* scriptExecutionContext() const final { return ActiveDOMObject::scriptExecutionContext(); }
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::ServiceWorkerContainer; }
    void refEventTarget() final;
    void derefEventTarget() final;

    // ActiveDOMObject.
    void stop() final;

    void notifyRegistrationIsSettled(const ServiceWorkerRegistrationKey&);

    std::unique_ptr<ReadyPromise> m_readyPromise;

    NavigatorBase& m_navigator;

    RefPtr<SWClientConnection> m_swConnection;

    struct OngoingJob {
        std::unique_ptr<ServiceWorkerJob> job;
        RefPtr<PendingActivity<ServiceWorkerContainer>> pendingActivity;
    };
    HashMap<ServiceWorkerJobIdentifier, OngoingJob> m_jobMap;

    bool m_isStopped { false };
    HashMap<ServiceWorkerRegistrationIdentifier, WeakRef<ServiceWorkerRegistration, WeakPtrImplWithEventTargetData>> m_registrations;

#if ASSERT_ENABLED
    Ref<Thread> m_creationThread { Thread::current() };
#endif

    uint64_t m_lastOngoingSettledRegistrationIdentifier { 0 };
    HashMap<uint64_t, ServiceWorkerRegistrationKey> m_ongoingSettledRegistrations;
    bool m_shouldDeferMessageEvents { false };
    Vector<MessageEvent::MessageEventWithStrongData> m_deferredMessageEvents;
};

} // namespace WebCore
