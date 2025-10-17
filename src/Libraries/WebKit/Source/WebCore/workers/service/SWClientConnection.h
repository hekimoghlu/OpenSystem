/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#include "BackgroundFetchRecordIdentifier.h"
#include "ExceptionOr.h"
#include "NavigationPreloadState.h"
#include "NotificationData.h"
#include "PushPermissionState.h"
#include "PushSubscriptionData.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerJob.h"
#include "ServiceWorkerTypes.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class ResourceError;
class ResourceResponse;
class SecurityOrigin;
class ScriptExecutionContext;
class SerializedScriptValue;
class ServiceWorkerContainer;
class ServiceWorkerRegistration;
enum class ServiceWorkerRegistrationState : uint8_t;
enum class ServiceWorkerState : uint8_t;
enum class ShouldNotifyWhenResolved : bool;
struct BackgroundFetchInformation;
struct BackgroundFetchOptions;
struct BackgroundFetchRecordInformation;
struct BackgroundFetchRequest;
struct CacheQueryOptions;
struct CookieChangeSubscription;
struct ClientOrigin;
struct ExceptionData;
struct MessageWithMessagePorts;
struct NotificationData;
struct RetrieveRecordsOptions;
struct ServiceWorkerClientData;
struct ServiceWorkerClientPendingMessage;
struct ServiceWorkerData;
struct ServiceWorkerRegistrationData;
struct WorkerFetchResult;

class SWClientConnection : public RefCounted<SWClientConnection> {
public:
    WEBCORE_EXPORT virtual ~SWClientConnection();

    static Ref<SWClientConnection> fromScriptExecutionContext(ScriptExecutionContext&);

    using RegistrationCallback = CompletionHandler<void(std::optional<ServiceWorkerRegistrationData>&&)>;
    virtual void matchRegistration(SecurityOriginData&& topOrigin, const URL& clientURL, RegistrationCallback&&) = 0;

    using GetRegistrationsCallback = CompletionHandler<void(Vector<ServiceWorkerRegistrationData>&&)>;
    virtual void getRegistrations(SecurityOriginData&& topOrigin, const URL& clientURL, GetRegistrationsCallback&&) = 0;

    using WhenRegistrationReadyCallback = Function<void(ServiceWorkerRegistrationData&&)>;
    virtual void whenRegistrationReady(const SecurityOriginData& topOrigin, const URL& clientURL, WhenRegistrationReadyCallback&&) = 0;

    virtual void addServiceWorkerRegistrationInServer(ServiceWorkerRegistrationIdentifier) = 0;
    virtual void removeServiceWorkerRegistrationInServer(ServiceWorkerRegistrationIdentifier) = 0;
    virtual void scheduleUnregisterJobInServer(ServiceWorkerRegistrationIdentifier, ServiceWorkerOrClientIdentifier, CompletionHandler<void(ExceptionOr<bool>&&)>&&) = 0;

    WEBCORE_EXPORT virtual void scheduleJob(ServiceWorkerOrClientIdentifier, const ServiceWorkerJobData&);

    virtual void didResolveRegistrationPromise(const ServiceWorkerRegistrationKey&) = 0;

    virtual void postMessageToServiceWorker(ServiceWorkerIdentifier destination, MessageWithMessagePorts&&, const ServiceWorkerOrClientIdentifier& source) = 0;

    virtual SWServerConnectionIdentifier serverConnectionIdentifier() const = 0;
    virtual bool mayHaveServiceWorkerRegisteredForOrigin(const SecurityOriginData&) const = 0;

    virtual void registerServiceWorkerClient(const ClientOrigin&, ServiceWorkerClientData&&, const std::optional<ServiceWorkerRegistrationIdentifier>&, String&& userAgent) = 0;
    virtual void unregisterServiceWorkerClient(ScriptExecutionContextIdentifier) = 0;

    virtual void finishFetchingScriptInServer(const ServiceWorkerJobDataIdentifier&, ServiceWorkerRegistrationKey&&, WorkerFetchResult&&) = 0;

    virtual void storeRegistrationsOnDiskForTesting(CompletionHandler<void()>&& callback) { callback(); }
    virtual void whenServiceWorkerIsTerminatedForTesting(ServiceWorkerIdentifier, CompletionHandler<void()>&& callback) { callback(); }
    
    using SubscribeToPushServiceCallback = CompletionHandler<void(ExceptionOr<PushSubscriptionData>&&)>;
    virtual void subscribeToPushService(ServiceWorkerRegistrationIdentifier, const Vector<uint8_t>& applicationServerKey, SubscribeToPushServiceCallback&&) = 0;
    
    using UnsubscribeFromPushServiceCallback = CompletionHandler<void(ExceptionOr<bool>&&)>;
    virtual void unsubscribeFromPushService(ServiceWorkerRegistrationIdentifier, PushSubscriptionIdentifier, UnsubscribeFromPushServiceCallback&&) = 0;
    
    using GetPushSubscriptionCallback = CompletionHandler<void(ExceptionOr<std::optional<PushSubscriptionData>>&&)>;
    virtual void getPushSubscription(ServiceWorkerRegistrationIdentifier, GetPushSubscriptionCallback&&) = 0;

    using GetPushPermissionStateCallback = CompletionHandler<void(ExceptionOr<PushPermissionState>&&)>;
    virtual void getPushPermissionState(ServiceWorkerRegistrationIdentifier, GetPushPermissionStateCallback&&) = 0;

#if ENABLE(NOTIFICATION_EVENT)
    using GetNotificationsCallback = CompletionHandler<void(ExceptionOr<Vector<NotificationData>>&&)>;
    virtual void getNotifications(const URL&, const String&, GetNotificationsCallback&&) = 0;
#endif

    using ExceptionOrVoidCallback = CompletionHandler<void(ExceptionOr<void>&&)>;
    virtual void enableNavigationPreload(ServiceWorkerRegistrationIdentifier, ExceptionOrVoidCallback&&) = 0;
    virtual void disableNavigationPreload(ServiceWorkerRegistrationIdentifier, ExceptionOrVoidCallback&&) = 0;
    virtual void setNavigationPreloadHeaderValue(ServiceWorkerRegistrationIdentifier, String&&, ExceptionOrVoidCallback&&) = 0;
    using ExceptionOrNavigationPreloadStateCallback = CompletionHandler<void(ExceptionOr<NavigationPreloadState>&&)>;
    virtual void getNavigationPreloadState(ServiceWorkerRegistrationIdentifier, ExceptionOrNavigationPreloadStateCallback&&) = 0;

    WEBCORE_EXPORT void registerServiceWorkerClients();
    bool isClosed() const { return m_isClosed; }

    using ExceptionOrBackgroundFetchInformationCallback = CompletionHandler<void(ExceptionOr<std::optional<BackgroundFetchInformation>>&&)>;
    virtual void startBackgroundFetch(ServiceWorkerRegistrationIdentifier, const String&, Vector<BackgroundFetchRequest>&&, BackgroundFetchOptions&&, ExceptionOrBackgroundFetchInformationCallback&&) = 0;
    virtual void backgroundFetchInformation(ServiceWorkerRegistrationIdentifier, const String&, ExceptionOrBackgroundFetchInformationCallback&&) = 0;
    using BackgroundFetchIdentifiersCallback = CompletionHandler<void(Vector<String>&&)>;
    virtual void backgroundFetchIdentifiers(ServiceWorkerRegistrationIdentifier, BackgroundFetchIdentifiersCallback&&) = 0;
    using AbortBackgroundFetchCallback = CompletionHandler<void(bool)>;
    virtual void abortBackgroundFetch(ServiceWorkerRegistrationIdentifier, const String&, AbortBackgroundFetchCallback&&) = 0;
    using MatchBackgroundFetchCallback = CompletionHandler<void(Vector<BackgroundFetchRecordInformation>&&)>;
    virtual void matchBackgroundFetch(ServiceWorkerRegistrationIdentifier, const String&, RetrieveRecordsOptions&&, MatchBackgroundFetchCallback&&) = 0;
    using RetrieveRecordResponseCallback = CompletionHandler<void(ExceptionOr<ResourceResponse>&&)>;
    virtual void retrieveRecordResponse(BackgroundFetchRecordIdentifier, RetrieveRecordResponseCallback&&) = 0;
    using RetrieveRecordResponseBodyCallback = Function<void(Expected<RefPtr<SharedBuffer>, ResourceError>&&)>;
    virtual void retrieveRecordResponseBody(BackgroundFetchRecordIdentifier, RetrieveRecordResponseBodyCallback&&) = 0;

    virtual void addCookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, Vector<CookieChangeSubscription>&&, ExceptionOrVoidCallback&&) = 0;
    virtual void removeCookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, Vector<CookieChangeSubscription>&&, ExceptionOrVoidCallback&&) = 0;
    using ExceptionOrCookieChangeSubscriptionsCallback = CompletionHandler<void(ExceptionOr<Vector<CookieChangeSubscription>>&&)>;
    virtual void cookieChangeSubscriptions(ServiceWorkerRegistrationIdentifier, ExceptionOrCookieChangeSubscriptionsCallback&&) = 0;

protected:
    WEBCORE_EXPORT SWClientConnection();

    WEBCORE_EXPORT void jobRejectedInServer(ServiceWorkerJobIdentifier, ExceptionData&&);
    WEBCORE_EXPORT void registrationJobResolvedInServer(ServiceWorkerJobIdentifier, ServiceWorkerRegistrationData&&, ShouldNotifyWhenResolved);
    WEBCORE_EXPORT void startScriptFetchForServer(ServiceWorkerJobIdentifier, ServiceWorkerRegistrationKey&&, FetchOptions::Cache);
    WEBCORE_EXPORT void postMessageToServiceWorkerClient(ScriptExecutionContextIdentifier destinationContextIdentifier, MessageWithMessagePorts&&, ServiceWorkerData&& source, String&& sourceOrigin);
    WEBCORE_EXPORT void updateRegistrationState(ServiceWorkerRegistrationIdentifier, ServiceWorkerRegistrationState, const std::optional<ServiceWorkerData>&);
    WEBCORE_EXPORT void updateWorkerState(ServiceWorkerIdentifier, ServiceWorkerState);
    WEBCORE_EXPORT void fireUpdateFoundEvent(ServiceWorkerRegistrationIdentifier);
    WEBCORE_EXPORT void setRegistrationLastUpdateTime(ServiceWorkerRegistrationIdentifier, WallTime);
    WEBCORE_EXPORT void setRegistrationUpdateViaCache(ServiceWorkerRegistrationIdentifier, ServiceWorkerUpdateViaCache);
    WEBCORE_EXPORT void notifyClientsOfControllerChange(const HashSet<ScriptExecutionContextIdentifier>& contextIdentifiers, std::optional<ServiceWorkerData>&& newController);
    WEBCORE_EXPORT void updateBackgroundFetchRegistration(const BackgroundFetchInformation&);

    WEBCORE_EXPORT void clearPendingJobs();
    void setIsClosed() { m_isClosed = true; }

private:
    virtual void scheduleJobInServer(const ServiceWorkerJobData&) = 0;

    enum class IsJobComplete : bool { No, Yes };
    bool postTaskForJob(ServiceWorkerJobIdentifier, IsJobComplete, Function<void(ServiceWorkerJob&)>&&);

    bool m_isClosed { false };
    HashMap<ServiceWorkerJobIdentifier, ServiceWorkerOrClientIdentifier> m_scheduledJobSources;
};

} // namespace WebCore
