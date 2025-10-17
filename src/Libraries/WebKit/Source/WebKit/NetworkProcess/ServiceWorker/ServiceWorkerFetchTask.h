/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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

#include "DownloadID.h"
#include <WebCore/FetchIdentifier.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <WebCore/ServiceWorkerTypes.h>
#include <WebCore/Timer.h>
#include <pal/SessionID.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
class SWServerRegistration;
}

namespace IPC {
class Connection;
class Decoder;
class FormDataReference;
class SharedBufferReference;
}

namespace WebCore {
class NetworkLoadMetrics;
}

namespace WebKit {
class DownloadManager;
class NetworkResourceLoader;
class NetworkSession;
class ServiceWorkerNavigationPreloader;
class WebSWServerConnection;
class WebSWServerToContextConnection;

class ServiceWorkerFetchTask : public RefCountedAndCanMakeWeakPtr<ServiceWorkerFetchTask> {
    WTF_MAKE_TZONE_ALLOCATED(ServiceWorkerFetchTask);
public:
    static RefPtr<ServiceWorkerFetchTask> fromNavigationPreloader(WebSWServerConnection&, NetworkResourceLoader&, const WebCore::ResourceRequest&, NetworkSession*);

    static Ref<ServiceWorkerFetchTask> create(WebSWServerConnection&, NetworkResourceLoader&, WebCore::ResourceRequest&&, WebCore::SWServerConnectionIdentifier, WebCore::ServiceWorkerIdentifier, WebCore::SWServerRegistration&, NetworkSession*, bool isWorkerReady);
    static Ref<ServiceWorkerFetchTask> create(WebSWServerConnection&, NetworkResourceLoader&, std::unique_ptr<ServiceWorkerNavigationPreloader>&&);

    ~ServiceWorkerFetchTask();

    void start(WebSWServerToContextConnection&);
    void cancelFromClient();
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    void continueDidReceiveFetchResponse();
    void continueFetchTaskWith(WebCore::ResourceRequest&&);

    WebCore::FetchIdentifier fetchIdentifier() const { return m_fetchIdentifier; }
    std::optional<WebCore::ServiceWorkerIdentifier> serviceWorkerIdentifier() const { return m_serviceWorkerIdentifier; }

    WebCore::ResourceRequest takeRequest() { return WTFMove(m_currentRequest); }

    void cannotHandle();
    void contextClosed();

    bool convertToDownload(DownloadManager&, DownloadID, const WebCore::ResourceRequest&, const WebCore::ResourceResponse&);

    MonotonicTime startTime() const;

private:
    ServiceWorkerFetchTask(WebSWServerConnection&, NetworkResourceLoader&, WebCore::ResourceRequest&&, WebCore::SWServerConnectionIdentifier, WebCore::ServiceWorkerIdentifier, WebCore::SWServerRegistration&, NetworkSession*, bool isWorkerReady);
    ServiceWorkerFetchTask(WebSWServerConnection&, NetworkResourceLoader&, std::unique_ptr<ServiceWorkerNavigationPreloader>&&);

    enum class ShouldSetSource : bool { No, Yes };
    void didReceiveRedirectResponse(WebCore::ResourceResponse&&);
    void didReceiveResponse(WebCore::ResourceResponse&&, bool needsContinueDidReceiveResponseMessage);
    void didReceiveData(const IPC::SharedBufferReference&, uint64_t encodedDataLength);
    void didReceiveDataFromPreloader(const WebCore::FragmentedSharedBuffer&, uint64_t encodedDataLength);
    void didReceiveFormData(const IPC::FormDataReference&);
    void didFinish(const WebCore::NetworkLoadMetrics&);
    void didFail(const WebCore::ResourceError&);
    void didNotHandle();
    void usePreload();

    void processRedirectResponse(WebCore::ResourceResponse&&, ShouldSetSource);
    void processResponse(WebCore::ResourceResponse&&, bool needsContinueDidReceiveResponseMessage, ShouldSetSource);

    void startFetch();

    void timeoutTimerFired();
    void softUpdateIfNeeded();
    void loadResponseFromPreloader();
    void loadBodyFromPreloader();
    void cancelPreloadIfNecessary();
    NetworkSession* session();
    void preloadResponseIsReady();

    void workerClosed();

    RefPtr<IPC::Connection> serviceWorkerConnection();
    template<typename Message> bool sendToClient(Message&&);

    RefPtr<NetworkResourceLoader> protectedLoader() const;
    void sendNavigationPreloadUpdate();

    WeakPtr<WebSWServerConnection> m_swServerConnection;
    WeakPtr<NetworkResourceLoader> m_loader;
    WeakPtr<WebSWServerToContextConnection> m_serviceWorkerConnection;
    WebCore::FetchIdentifier m_fetchIdentifier;
    Markable<WebCore::SWServerConnectionIdentifier> m_serverConnectionIdentifier;
    Markable<WebCore::ServiceWorkerIdentifier> m_serviceWorkerIdentifier;
    WebCore::ResourceRequest m_currentRequest;
    std::unique_ptr<WebCore::Timer> m_timeoutTimer;
    Markable<WebCore::ServiceWorkerRegistrationIdentifier> m_serviceWorkerRegistrationIdentifier;
    std::unique_ptr<ServiceWorkerNavigationPreloader> m_preloader;
    bool m_wasHandled { false };
    bool m_isDone { false };
    bool m_shouldSoftUpdate { false };
    bool m_isLoadingFromPreloader { false };
};

}
