/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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

#include "SWClientConnection.h"
#include <wtf/Forward.h>
#include <wtf/ObjectIdentifier.h>

namespace WebCore {

class WorkerGlobalScope;
class WorkerThread;

struct CookieChangeSubscription;

class WorkerSWClientConnection final : public SWClientConnection {
public:
    static Ref<WorkerSWClientConnection> create(WorkerGlobalScope& scope) { return adoptRef(*new WorkerSWClientConnection { scope }); }
    ~WorkerSWClientConnection();

    void registerServiceWorkerClient(const ClientOrigin&, ServiceWorkerClientData&&, const std::optional<ServiceWorkerRegistrationIdentifier>&, String&& userAgent) final;
    void unregisterServiceWorkerClient(ScriptExecutionContextIdentifier) final;

private:
    explicit WorkerSWClientConnection(WorkerGlobalScope&);

    void matchRegistration(SecurityOriginData&& topOrigin, const URL& clientURL, RegistrationCallback&&) final;
    void getRegistrations(SecurityOriginData&& topOrigin, const URL& clientURL, GetRegistrationsCallback&&) final;
    void whenRegistrationReady(const SecurityOriginData& topOrigin, const URL& clientURL, WhenRegistrationReadyCallback&&) final;
    void addServiceWorkerRegistrationInServer(ServiceWorkerRegistrationIdentifier) final;
    void removeServiceWorkerRegistrationInServer(ServiceWorkerRegistrationIdentifier) final;
    void didResolveRegistrationPromise(const ServiceWorkerRegistrationKey&) final;
    void postMessageToServiceWorker(ServiceWorkerIdentifier destination, MessageWithMessagePorts&&, const ServiceWorkerOrClientIdentifier& source) final;
    SWServerConnectionIdentifier serverConnectionIdentifier() const final;
    bool mayHaveServiceWorkerRegisteredForOrigin(const SecurityOriginData&) const final;
    void finishFetchingScriptInServer(const ServiceWorkerJobDataIdentifier&, ServiceWorkerRegistrationKey&&, WorkerFetchResult&&) final;
    void scheduleJobInServer(const ServiceWorkerJobData&) final;
    void scheduleJob(ServiceWorkerOrClientIdentifier, const ServiceWorkerJobData&) final;
    void scheduleUnregisterJobInServer(ServiceWorkerRegistrationIdentifier, ServiceWorkerOrClientIdentifier, CompletionHandler<void(ExceptionOr<bool>&&)>&&) final;
    void subscribeToPushService(ServiceWorkerRegistrationIdentifier, const Vector<uint8_t>& applicationServerKey, SubscribeToPushServiceCallback&&) final;
    void unsubscribeFromPushService(ServiceWorkerRegistrationIdentifier, PushSubscriptionIdentifier, UnsubscribeFromPushServiceCallback&&) final;
    void getPushSubscription(ServiceWorkerRegistrationIdentifier, GetPushSubscriptionCallback&&) final;
    void getPushPermissionState(ServiceWorkerRegistrationIdentifier, GetPushPermissionStateCallback&&) final;
#if ENABLE(NOTIFICATION_EVENT)
    void getNotifications(const URL&, const String&, GetNotificationsCallback&&) final;
#endif

    void enableNavigationPreload(ServiceWorkerRegistrationIdentifier, ExceptionOrVoidCallback&&) final;
    void disableNavigationPreload(ServiceWorkerRegistrationIdentifier, ExceptionOrVoidCallback&&) final;
    void setNavigationPreloadHeaderValue(ServiceWorkerRegistrationIdentifier, String&&, ExceptionOrVoidCallback&&) final;
    void getNavigationPreloadState(ServiceWorkerRegistrationIdentifier, ExceptionOrNavigationPreloadStateCallback&&) final;

    void startBackgroundFetch(ServiceWorkerRegistrationIdentifier, const String&, Vector<BackgroundFetchRequest>&&, BackgroundFetchOptions&&, ExceptionOrBackgroundFetchInformationCallback&&) final;
    void backgroundFetchInformation(ServiceWorkerRegistrationIdentifier, const String&, ExceptionOrBackgroundFetchInformationCallback&&) final;
    void backgroundFetchIdentifiers(ServiceWorkerRegistrationIdentifier, BackgroundFetchIdentifiersCallback&&) final;
    void abortBackgroundFetch(ServiceWorkerRegistrationIdentifier, const String&, AbortBackgroundFetchCallback&&) final;
    void matchBackgroundFetch(ServiceWorkerRegistrationIdentifier, const String&, RetrieveRecordsOptions&&, MatchBackgroundFetchCallback&&) final;
    void retrieveRecordResponse(BackgroundFetchRecordIdentifier, RetrieveRecordResponseCallback&&) final;
    void retrieveRecordResponseBody(BackgroundFetchRecordIdentifier, RetrieveRecordResponseBodyCallback&&) final;

    void addCookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, Vector<CookieChangeSubscription>&&, ExceptionOrVoidCallback&&) final;
    void removeCookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, Vector<CookieChangeSubscription>&&, ExceptionOrVoidCallback&&) final;
    void cookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, ExceptionOrCookieChangeSubscriptionsCallback&&) final;

    Ref<WorkerThread> m_thread;

    struct SWClientRequestIdentifierType;
    using SWClientRequestIdentifier = AtomicObjectIdentifier<SWClientRequestIdentifierType>;

    HashMap<SWClientRequestIdentifier, RegistrationCallback> m_matchRegistrationRequests;
    HashMap<SWClientRequestIdentifier, GetRegistrationsCallback> m_getRegistrationsRequests;
    HashMap<SWClientRequestIdentifier, WhenRegistrationReadyCallback> m_whenRegistrationReadyRequests;
    HashMap<SWClientRequestIdentifier, CompletionHandler<void(ExceptionOr<bool>&&)>> m_unregisterRequests;
    HashMap<SWClientRequestIdentifier, SubscribeToPushServiceCallback> m_subscribeToPushServiceRequests;
    HashMap<SWClientRequestIdentifier, UnsubscribeFromPushServiceCallback> m_unsubscribeFromPushServiceRequests;
    HashMap<SWClientRequestIdentifier, GetPushSubscriptionCallback> m_getPushSubscriptionRequests;
    HashMap<SWClientRequestIdentifier, GetPushPermissionStateCallback> m_getPushPermissionStateCallbacks;
    HashMap<SWClientRequestIdentifier, ExceptionOrVoidCallback> m_voidCallbacks;
    HashMap<SWClientRequestIdentifier, ExceptionOrNavigationPreloadStateCallback> m_navigationPreloadStateCallbacks;
#if ENABLE(NOTIFICATION_EVENT)
    HashMap<SWClientRequestIdentifier, GetNotificationsCallback> m_getNotificationsCallbacks;
#endif
    HashMap<SWClientRequestIdentifier, ExceptionOrBackgroundFetchInformationCallback> m_backgroundFetchInformationCallbacks;
    HashMap<SWClientRequestIdentifier, BackgroundFetchIdentifiersCallback> m_backgroundFetchIdentifiersCallbacks;
    HashMap<SWClientRequestIdentifier, AbortBackgroundFetchCallback> m_abortBackgroundFetchCallbacks;
    HashMap<SWClientRequestIdentifier, MatchBackgroundFetchCallback> m_matchBackgroundFetchCallbacks;
    HashMap<SWClientRequestIdentifier, RetrieveRecordResponseCallback> m_retrieveRecordResponseCallbacks;
    HashMap<SWClientRequestIdentifier, RetrieveRecordResponseBodyCallback> m_retrieveRecordResponseBodyCallbacks;
    HashMap<SWClientRequestIdentifier, ExceptionOrCookieChangeSubscriptionsCallback> m_cookieChangeSubscriptionsCallback;
};

} // namespace WebCore
