/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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

#include "Connection.h"
#include "IdentifierTypes.h"
#include "MessageReceiver.h"
#include "NavigatingToAppBoundDomain.h"
#include "UserContentControllerIdentifier.h"
#include "WebPageProxyIdentifier.h"
#include "WebPreferencesStore.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/BackgroundFetchFailureReason.h>
#include <WebCore/SWContextManager.h>
#include <WebCore/ServiceWorkerClientData.h>
#include <WebCore/ServiceWorkerTypes.h>
#include <WebCore/Site.h>
#include <wtf/URLHash.h>

namespace IPC {
class FormDataReference;
}

namespace WebCore {
class ResourceRequest;
class Site;

enum class WorkerThreadMode : bool;

struct FetchOptions;
struct NotificationPayload;
struct ServiceWorkerContextData;
}

namespace WebKit {

class RemoteWorkerFrameLoaderClient;
class WebServiceWorkerFetchTaskClient;
class WebUserContentController;
struct RemoteWorkerInitializationData;

class WebSWContextManagerConnection final : public WebCore::SWContextManager::Connection, public IPC::WorkQueueMessageReceiver {
public:
    static Ref<WebSWContextManagerConnection> create(Ref<IPC::Connection>&& connection, WebCore::Site&& site, std::optional<WebCore::ScriptExecutionContextIdentifier> serviceWorkerPageIdentifier, PageGroupIdentifier pageGroupID, WebPageProxyIdentifier webPageProxyID, WebCore::PageIdentifier pageID, const WebPreferencesStore& store, RemoteWorkerInitializationData&& initializationData) { return adoptRef(*new WebSWContextManagerConnection(WTFMove(connection), WTFMove(site), serviceWorkerPageIdentifier, pageGroupID, webPageProxyID, pageID, store, WTFMove(initializationData))); }
    ~WebSWContextManagerConnection();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    WebCore::PageIdentifier pageIdentifier() const final { return m_pageID; }

    void ref() const final { return IPC::WorkQueueMessageReceiver::ref(); }
    void deref() const final { return IPC::WorkQueueMessageReceiver::deref(); }

private:
    WebSWContextManagerConnection(Ref<IPC::Connection>&&, WebCore::Site&&, std::optional<WebCore::ScriptExecutionContextIdentifier> serviceWorkerPageIdentifier, PageGroupIdentifier, WebPageProxyIdentifier, WebCore::PageIdentifier, const WebPreferencesStore&, RemoteWorkerInitializationData&&);

    // WebCore::SWContextManager::Connection.
    void establishConnection(CompletionHandler<void()>&&) final;
    void postMessageToServiceWorkerClient(const WebCore::ScriptExecutionContextIdentifier& destinationIdentifier, const WebCore::MessageWithMessagePorts&, WebCore::ServiceWorkerIdentifier sourceIdentifier, const String& sourceOrigin) final;
    void didFinishInstall(std::optional<WebCore::ServiceWorkerJobDataIdentifier>, WebCore::ServiceWorkerIdentifier, bool wasSuccessful) final;
    void didFinishActivation(WebCore::ServiceWorkerIdentifier) final;
    void setServiceWorkerHasPendingEvents(WebCore::ServiceWorkerIdentifier, bool) final;
    void workerTerminated(WebCore::ServiceWorkerIdentifier) final;
    void findClientByVisibleIdentifier(WebCore::ServiceWorkerIdentifier, const String&, FindClientByIdentifierCallback&&) final;
    void matchAll(WebCore::ServiceWorkerIdentifier, const WebCore::ServiceWorkerClientQueryOptions&, WebCore::ServiceWorkerClientsMatchAllCallback&&) final;
    void claim(WebCore::ServiceWorkerIdentifier, CompletionHandler<void(WebCore::ExceptionOr<void>&&)>&&) final;
    void focus(WebCore::ScriptExecutionContextIdentifier, CompletionHandler<void(std::optional<WebCore::ServiceWorkerClientData>&&)>&&) final;
    void navigate(WebCore::ScriptExecutionContextIdentifier, WebCore::ServiceWorkerIdentifier, const URL&, NavigateCallback&&) final;
    void skipWaiting(WebCore::ServiceWorkerIdentifier, CompletionHandler<void()>&&) final;
    void setScriptResource(WebCore::ServiceWorkerIdentifier, const URL&, const WebCore::ServiceWorkerContextData::ImportedScript&) final;
    bool isThrottleable() const final;
    void didFailHeartBeatCheck(WebCore::ServiceWorkerIdentifier) final;
    void setAsInspected(WebCore::ServiceWorkerIdentifier, bool) final;
    void openWindow(WebCore::ServiceWorkerIdentifier, const URL&, OpenWindowCallback&&) final;
    void stop() final;
    void reportConsoleMessage(WebCore::ServiceWorkerIdentifier, MessageSource, MessageLevel, const String& message, unsigned long requestIdentifier) final;
    void removeNavigationFetch(WebCore::SWServerConnectionIdentifier, WebCore::FetchIdentifier) final;

    // IPC messages.
    void updatePreferencesStore(WebPreferencesStore&&);
    void serviceWorkerStarted(std::optional<WebCore::ServiceWorkerJobDataIdentifier>, WebCore::ServiceWorkerIdentifier, bool doesHandleFetch) final;
    void serviceWorkerFailedToStart(std::optional<WebCore::ServiceWorkerJobDataIdentifier>, WebCore::ServiceWorkerIdentifier, const String& exceptionMessage) final;
    void installServiceWorker(WebCore::ServiceWorkerContextData&&, WebCore::ServiceWorkerData&&, String&& userAgent, WebCore::WorkerThreadMode, WebCore::ServiceWorkerIsInspectable, OptionSet<WebCore::AdvancedPrivacyProtections>);
    void updateAppInitiatedValue(WebCore::ServiceWorkerIdentifier, WebCore::LastNavigationWasAppInitiated);
    void startFetch(WebCore::SWServerConnectionIdentifier, WebCore::ServiceWorkerIdentifier, WebCore::FetchIdentifier, WebCore::ResourceRequest&&, WebCore::FetchOptions&&, IPC::FormDataReference&&, String&& referrer, bool isServiceWorkerNavigationPreloadEnabled, String&& clientIdentifier, String&& resultingClientIdentifier);
    void cancelFetch(WebCore::SWServerConnectionIdentifier, WebCore::ServiceWorkerIdentifier, WebCore::FetchIdentifier);
    void continueDidReceiveFetchResponse(WebCore::SWServerConnectionIdentifier, WebCore::ServiceWorkerIdentifier, WebCore::FetchIdentifier);
    void postMessageToServiceWorker(WebCore::ServiceWorkerIdentifier destinationIdentifier, WebCore::MessageWithMessagePorts&&, WebCore::ServiceWorkerOrClientData&& sourceData);
    void fireInstallEvent(WebCore::ServiceWorkerIdentifier);
    void fireActivateEvent(WebCore::ServiceWorkerIdentifier);
    void firePushEvent(WebCore::ServiceWorkerIdentifier, std::optional<std::span<const uint8_t>>, std::optional<WebCore::NotificationPayload>&&, CompletionHandler<void(bool, std::optional<WebCore::NotificationPayload>&&)>&&);
    void fireNotificationEvent(WebCore::ServiceWorkerIdentifier, WebCore::NotificationData&&, WebCore::NotificationEventType, CompletionHandler<void(bool)>&&);
    void fireBackgroundFetchEvent(WebCore::ServiceWorkerIdentifier, WebCore::BackgroundFetchInformation&&, CompletionHandler<void(bool)>&&);
    void fireBackgroundFetchClickEvent(WebCore::ServiceWorkerIdentifier, WebCore::BackgroundFetchInformation&&, CompletionHandler<void(bool)>&&);
    void terminateWorker(WebCore::ServiceWorkerIdentifier);
#if ENABLE(SHAREABLE_RESOURCE) && PLATFORM(COCOA)
    void didSaveScriptsToDisk(WebCore::ServiceWorkerIdentifier, WebCore::ScriptBuffer&&, HashMap<URL, WebCore::ScriptBuffer>&& importedScripts);
#endif
    void setUserAgent(String&& userAgent);
    void close();
    void setThrottleState(bool isThrottleable);
    void setInspectable(WebCore::ServiceWorkerIsInspectable);
    void convertFetchToDownload(WebCore::SWServerConnectionIdentifier, WebCore::ServiceWorkerIdentifier, WebCore::FetchIdentifier);
    void cancelFetchDownload(WebCore::ServiceWorkerIdentifier, WebCore::FetchIdentifier);
    void navigationPreloadIsReady(WebCore::SWServerConnectionIdentifier, WebCore::ServiceWorkerIdentifier, WebCore::FetchIdentifier, WebCore::ResourceResponse&&);
    void navigationPreloadFailed(WebCore::SWServerConnectionIdentifier, WebCore::ServiceWorkerIdentifier, WebCore::FetchIdentifier, WebCore::ResourceError&&);

    void updateRegistrationState(WebCore::ServiceWorkerRegistrationIdentifier, WebCore::ServiceWorkerRegistrationState, const std::optional<WebCore::ServiceWorkerData>&);
    void updateWorkerState(WebCore::ServiceWorkerIdentifier, WebCore::ServiceWorkerState);
    void fireUpdateFoundEvent(WebCore::ServiceWorkerRegistrationIdentifier);
    void setRegistrationLastUpdateTime(WebCore::ServiceWorkerRegistrationIdentifier, WallTime);
    void setRegistrationUpdateViaCache(WebCore::ServiceWorkerRegistrationIdentifier, WebCore::ServiceWorkerUpdateViaCache);

    Ref<IPC::Connection> m_connectionToNetworkProcess;
    const WebCore::Site m_site;
    std::optional<WebCore::ScriptExecutionContextIdentifier> m_serviceWorkerPageIdentifier;
    PageGroupIdentifier m_pageGroupID;
    WebPageProxyIdentifier m_webPageProxyID;
    WebCore::PageIdentifier m_pageID;

    HashSet<std::unique_ptr<RemoteWorkerFrameLoaderClient>> m_loaders;
    String m_userAgent;
    bool m_isThrottleable { true };
    Ref<WebUserContentController> m_userContentController;
    std::optional<WebPreferencesStore> m_preferencesStore;
    Ref<WorkQueue> m_queue;

    using FetchKey = std::pair<WebCore::SWServerConnectionIdentifier, WebCore::FetchIdentifier>;
    HashMap<FetchKey, Ref<WebServiceWorkerFetchTaskClient>> m_ongoingNavigationFetchTasks WTF_GUARDED_BY_CAPABILITY(m_queue.get());
};

} // namespace WebKit
