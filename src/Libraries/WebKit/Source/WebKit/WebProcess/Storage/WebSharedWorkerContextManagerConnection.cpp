/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include "config.h"
#include "WebSharedWorkerContextManagerConnection.h"

#include "Logging.h"
#include "NetworkConnectionToWebProcessMessages.h"
#include "RemoteWebLockRegistry.h"
#include "RemoteWorkerFrameLoaderClient.h"
#include "RemoteWorkerInitializationData.h"
#include "RemoteWorkerLibWebRTCProvider.h"
#include "WebBadgeClient.h"
#include "WebBroadcastChannelRegistry.h"
#include "WebCacheStorageProvider.h"
#include "WebCompiledContentRuleListData.h"
#include "WebDatabaseProvider.h"
#include "WebPage.h"
#include "WebPreferencesKeys.h"
#include "WebProcess.h"
#include "WebSharedWorkerServerToContextConnectionMessages.h"
#include "WebSocketProvider.h"
#include "WebStorageProvider.h"
#include "WebUserContentController.h"
#include "WebWorkerClient.h"
#include <WebCore/EmptyClients.h>
#include <WebCore/Page.h>
#include <WebCore/PageConfiguration.h>
#include <WebCore/RemoteFrameClient.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <WebCore/SharedWorkerContextManager.h>
#include <WebCore/SharedWorkerThread.h>
#include <WebCore/SharedWorkerThreadProxy.h>
#include <WebCore/UserAgent.h>
#include <WebCore/WorkerFetchResult.h>
#include <WebCore/WorkerInitializationData.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSharedWorkerContextManagerConnection);

Ref<WebSharedWorkerContextManagerConnection> WebSharedWorkerContextManagerConnection::create(Ref<IPC::Connection>&& connectionToNetworkProcess, WebCore::Site&& site, PageGroupIdentifier pageGroupID, WebPageProxyIdentifier webPageProxyID, WebCore::PageIdentifier pageID, const WebPreferencesStore& preferencesStore, RemoteWorkerInitializationData&& initializationData)
{
    return adoptRef(*new WebSharedWorkerContextManagerConnection(WTFMove(connectionToNetworkProcess), WTFMove(site), pageGroupID, webPageProxyID, pageID, preferencesStore, WTFMove(initializationData)));
}

WebSharedWorkerContextManagerConnection::WebSharedWorkerContextManagerConnection(Ref<IPC::Connection>&& connectionToNetworkProcess, WebCore::Site&& site, PageGroupIdentifier pageGroupID, WebPageProxyIdentifier webPageProxyID, WebCore::PageIdentifier pageID, const WebPreferencesStore& preferencesStore, RemoteWorkerInitializationData&& initializationData)
    : m_connectionToNetworkProcess(WTFMove(connectionToNetworkProcess))
    , m_site(WTFMove(site))
    , m_pageGroupID(pageGroupID)
    , m_webPageProxyID(webPageProxyID)
    , m_pageID(pageID)
#if PLATFORM(COCOA)
    , m_userAgent(WebCore::standardUserAgentWithApplicationName({ }))
#else
    , m_userAgent(WebCore::standardUserAgent())
#endif
    , m_userContentController(WebUserContentController::getOrCreate(initializationData.userContentControllerIdentifier))
{
#if ENABLE(CONTENT_EXTENSIONS)
    m_userContentController->addContentRuleLists(WTFMove(initializationData.contentRuleLists));
#endif

    updatePreferencesStore(preferencesStore);
    WebProcess::singleton().disableTermination();
}

WebSharedWorkerContextManagerConnection::~WebSharedWorkerContextManagerConnection() = default;

void WebSharedWorkerContextManagerConnection::establishConnection(CompletionHandler<void()>&& completionHandler)
{
    m_connectionToNetworkProcess->sendWithAsyncReply(Messages::NetworkConnectionToWebProcess::EstablishSharedWorkerContextConnection { m_webPageProxyID, m_site }, WTFMove(completionHandler), 0);
}

void WebSharedWorkerContextManagerConnection::postErrorToWorkerObject(WebCore::SharedWorkerIdentifier sharedWorkerIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent)
{
    m_connectionToNetworkProcess->send(Messages::WebSharedWorkerServerToContextConnection::PostErrorToWorkerObject { sharedWorkerIdentifier, errorMessage, lineNumber, columnNumber, sourceURL, isErrorEvent }, 0);
}

void WebSharedWorkerContextManagerConnection::updatePreferencesStore(const WebPreferencesStore& store)
{
    WebPage::updatePreferencesGenerated(store);
    m_preferencesStore = store;
}

void WebSharedWorkerContextManagerConnection::launchSharedWorker(WebCore::ClientOrigin&& origin, WebCore::SharedWorkerIdentifier sharedWorkerIdentifier, WebCore::WorkerOptions&& workerOptions, WebCore::WorkerFetchResult&& workerFetchResult, WebCore::WorkerInitializationData&& initializationData)
{
    RELEASE_LOG(SharedWorker, "WebSharedWorkerContextManagerConnection::launchSharedWorker: sharedWorkerIdentifier=%" PRIu64, sharedWorkerIdentifier.toUInt64());
    auto pageConfiguration = WebCore::pageConfigurationWithEmptyClients(m_pageID, WebProcess::singleton().sessionID());
    pageConfiguration.badgeClient = WebBadgeClient::create();
    pageConfiguration.databaseProvider = WebDatabaseProvider::getOrCreate(m_pageGroupID);
    pageConfiguration.socketProvider = WebSocketProvider::create(m_webPageProxyID);
    pageConfiguration.broadcastChannelRegistry = WebProcess::singleton().broadcastChannelRegistry();
    pageConfiguration.userContentProvider = m_userContentController;
#if ENABLE(WEB_RTC)
    pageConfiguration.webRTCProvider = makeUniqueRef<RemoteWorkerLibWebRTCProvider>();
#endif
    pageConfiguration.storageProvider = makeUniqueRef<WebStorageProvider>(WebProcess::singleton().mediaKeysStorageDirectory(), WebProcess::singleton().mediaKeysStorageSalt());

    pageConfiguration.mainFrameCreationParameters = WebCore::PageConfiguration::LocalMainFrameCreationParameters { CompletionHandler<UniqueRef<WebCore::LocalFrameLoaderClient>(WebCore::LocalFrame&, WebCore::FrameLoader&)> { [webPageProxyID = m_webPageProxyID, pageID = m_pageID, userAgent = m_userAgent] (auto&, auto& frameLoader) mutable {
        return makeUniqueRefWithoutRefCountedCheck<RemoteWorkerFrameLoaderClient>(frameLoader, webPageProxyID, pageID, userAgent);
    } }, WebCore::SandboxFlags { } };

    Ref page = WebCore::Page::create(WTFMove(pageConfiguration));
    if (m_preferencesStore) {
        WebPage::updateSettingsGenerated(*m_preferencesStore, page->settings());
        page->settings().setStorageBlockingPolicy(static_cast<WebCore::StorageBlockingPolicy>(m_preferencesStore->getUInt32ValueForKey(WebPreferencesKey::storageBlockingPolicyKey())));
    }
    if (WebProcess::singleton().isLockdownModeEnabled())
        WebPage::adjustSettingsForLockdownMode(page->settings(), m_preferencesStore ? &m_preferencesStore.value() : nullptr);

    if (!initializationData.userAgent.isEmpty())
        initializationData.userAgent = m_userAgent;

    if (!initializationData.clientIdentifier)
        initializationData.clientIdentifier = WebCore::ScriptExecutionContextIdentifier::generate();

    page->setupForRemoteWorker(workerFetchResult.responseURL, origin.topOrigin, workerFetchResult.referrerPolicy, initializationData.advancedPrivacyProtections);

    auto sharedWorkerThreadProxy = WebCore::SharedWorkerThreadProxy::create(Ref { page }, sharedWorkerIdentifier, origin, WTFMove(workerFetchResult), WTFMove(workerOptions), WTFMove(initializationData), WebProcess::singleton().cacheStorageProvider());

    Ref thread = sharedWorkerThreadProxy->thread();
    auto workerClient = WebWorkerClient::create(WTFMove(page), thread);
    thread->setWorkerClient(workerClient.moveToUniquePtr());

    WebCore::SharedWorkerContextManager::singleton().registerSharedWorkerThread(WTFMove(sharedWorkerThreadProxy));
}

void WebSharedWorkerContextManagerConnection::close()
{
    RELEASE_LOG(SharedWorker, "WebSharedWorkerContextManagerConnection::close: Shared worker process is requested to stop all shared workers (already stopped = %d)", isClosed());
    if (isClosed())
        return;

    setAsClosed();

    m_connectionToNetworkProcess->send(Messages::NetworkConnectionToWebProcess::CloseSharedWorkerContextConnection { }, 0);
    WebCore::SharedWorkerContextManager::singleton().stopAllSharedWorkers();
    WebProcess::singleton().enableTermination();
}

void WebSharedWorkerContextManagerConnection::sharedWorkerTerminated(WebCore::SharedWorkerIdentifier sharedWorkerIdentifier)
{
    RELEASE_LOG(SharedWorker, "WebSharedWorkerContextManagerConnection::sharedWorkerTerminated: sharedWorkerIdentifier=%" PRIu64, sharedWorkerIdentifier.toUInt64());
    m_connectionToNetworkProcess->send(Messages::WebSharedWorkerServerToContextConnection::SharedWorkerTerminated { sharedWorkerIdentifier }, 0);
}

} // namespace WebKit
