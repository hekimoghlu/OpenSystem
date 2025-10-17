/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#include "SharedWorkerThreadProxy.h"

#include "BadgeClient.h"
#include "CacheStorageProvider.h"
#include "Chrome.h"
#include "ErrorEvent.h"
#include "EventLoop.h"
#include "EventNames.h"
#include "FrameLoader.h"
#include "LoaderStrategy.h"
#include "LocalFrame.h"
#include "MessageEvent.h"
#include "MessagePort.h"
#include "Page.h"
#include "PlatformStrategies.h"
#include "RTCDataChannelRemoteHandlerConnection.h"
#include "SharedWorker.h"
#include "SharedWorkerContextManager.h"
#include "SharedWorkerGlobalScope.h"
#include "SharedWorkerThread.h"
#include "WebRTCProvider.h"
#include "WorkerClient.h"
#include "WorkerFetchResult.h"
#include "WorkerInitializationData.h"
#include "WorkerThread.h"
#include <JavaScriptCore/IdentifiersFactory.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

static HashMap<ScriptExecutionContextIdentifier, WeakRef<SharedWorkerThreadProxy>>& allSharedWorkerThreadProxies()
{
    static MainThreadNeverDestroyed<HashMap<ScriptExecutionContextIdentifier, WeakRef<SharedWorkerThreadProxy>>> map;
    return map;
}

static WorkerParameters generateWorkerParameters(const WorkerFetchResult& workerFetchResult, WorkerOptions&& workerOptions, WorkerInitializationData&& initializationData, Document& document)
{
    RELEASE_ASSERT(document.sessionID());
    return {
        workerFetchResult.responseURL,
        document.url(),
        workerOptions.name,
        makeString("sharedworker:"_s, Inspector::IdentifiersFactory::createIdentifier()),
        WTFMove(initializationData.userAgent),
        platformStrategies()->loaderStrategy()->isOnLine(),
        workerFetchResult.contentSecurityPolicy,
        false,
        workerFetchResult.crossOriginEmbedderPolicy,
        MonotonicTime::now(),
        parseReferrerPolicy(workerFetchResult.referrerPolicy, ReferrerPolicySource::HTTPHeader).value_or(ReferrerPolicy::EmptyString),
        workerOptions.type,
        workerOptions.credentials,
        document.settingsValues(),
        WorkerThreadMode::CreateNewThread,
        *document.sessionID(),
        WTFMove(initializationData.serviceWorkerData),
        *initializationData.clientIdentifier,
        document.advancedPrivacyProtections(),
        document.noiseInjectionHashSalt()
    };
}

SharedWorkerThreadProxy* SharedWorkerThreadProxy::byIdentifier(ScriptExecutionContextIdentifier identifier)
{
    return allSharedWorkerThreadProxies().get(identifier);
}

bool SharedWorkerThreadProxy::hasInstances()
{
    return !allSharedWorkerThreadProxies().isEmpty();
}

SharedWorkerThreadProxy::SharedWorkerThreadProxy(Ref<Page>&& page, SharedWorkerIdentifier sharedWorkerIdentifier, const ClientOrigin& clientOrigin, WorkerFetchResult&& workerFetchResult, WorkerOptions&& workerOptions, WorkerInitializationData&& initializationData, CacheStorageProvider& cacheStorageProvider)
    : m_page(WTFMove(page))
    , m_document(*m_page->localTopDocument())
    , m_contextIdentifier(*initializationData.clientIdentifier)
    , m_workerThread(SharedWorkerThread::create(sharedWorkerIdentifier, generateWorkerParameters(workerFetchResult, WTFMove(workerOptions), WTFMove(initializationData), m_document), WTFMove(workerFetchResult.script), *this, *this, *this, *this, WorkerThreadStartMode::Normal, clientOrigin.topOrigin.securityOrigin(), m_document->idbConnectionProxy(), m_document->socketProvider(), JSC::RuntimeFlags::createAllEnabled()))
    , m_cacheStorageProvider(cacheStorageProvider)
    , m_clientOrigin(clientOrigin)
{
    ASSERT(!allSharedWorkerThreadProxies().contains(m_contextIdentifier));
    allSharedWorkerThreadProxies().add(m_contextIdentifier, *this);

    static bool addedListener;
    if (!addedListener) {
        platformStrategies()->loaderStrategy()->addOnlineStateChangeListener(&networkStateChanged);
        addedListener = true;
    }

    if (auto workerClient = m_page->chrome().createWorkerClient(thread()))
        thread().setWorkerClient(WTFMove(workerClient));
}

SharedWorkerThreadProxy::~SharedWorkerThreadProxy()
{
    ASSERT(allSharedWorkerThreadProxies().contains(m_contextIdentifier));
    allSharedWorkerThreadProxies().remove(m_contextIdentifier);

    m_workerThread->clearProxies();
}

SharedWorkerIdentifier SharedWorkerThreadProxy::identifier() const
{
    return m_workerThread->identifier();
}

void SharedWorkerThreadProxy::notifyNetworkStateChange(bool isOnline)
{
    if (m_isTerminatingOrTerminated)
        return;

    postTaskForModeToWorkerOrWorkletGlobalScope([isOnline] (ScriptExecutionContext& context) {
        auto& globalScope = downcast<WorkerGlobalScope>(context);
        globalScope.setIsOnline(isOnline);
        globalScope.eventLoop().queueTask(TaskSource::DOMManipulation, [globalScope = Ref { globalScope }, isOnline] {
            globalScope->dispatchEvent(Event::create(isOnline ? eventNames().onlineEvent : eventNames().offlineEvent, Event::CanBubble::No, Event::IsCancelable::No));
        });
    }, WorkerRunLoop::defaultMode());
}

void SharedWorkerThreadProxy::postExceptionToWorkerObject(const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL)
{
    ASSERT(!isMainThread());
    if (!m_workerThread->isInStaticScriptEvaluation())
        return;

    callOnMainThread([sharedWorkerIdentifier = m_workerThread->identifier(), errorMessage = errorMessage.isolatedCopy(), lineNumber, columnNumber, sourceURL = sourceURL.isolatedCopy()] {
        bool isErrorEvent = true;
        if (RefPtr connection = SharedWorkerContextManager::singleton().connection())
            connection->postErrorToWorkerObject(sharedWorkerIdentifier, errorMessage, lineNumber, columnNumber, sourceURL, isErrorEvent);
    });
}

void SharedWorkerThreadProxy::reportErrorToWorkerObject(const String& errorMessage)
{
    ASSERT(!isMainThread());

    callOnMainThread([sharedWorkerIdentifier = m_workerThread->identifier(), errorMessage = errorMessage.isolatedCopy()] {
        bool isErrorEvent = false;
        if (RefPtr connection = SharedWorkerContextManager::singleton().connection())
            connection->postErrorToWorkerObject(sharedWorkerIdentifier, errorMessage, 0, 0, { }, isErrorEvent);
    });
}

RefPtr<CacheStorageConnection> SharedWorkerThreadProxy::createCacheStorageConnection()
{
    ASSERT(isMainThread());
    if (!m_cacheStorageConnection)
        m_cacheStorageConnection = m_cacheStorageProvider.createCacheStorageConnection();
    return m_cacheStorageConnection;
}

RefPtr<RTCDataChannelRemoteHandlerConnection> SharedWorkerThreadProxy::createRTCDataChannelRemoteHandlerConnection()
{
    ASSERT(isMainThread());
    return m_page->webRTCProvider().createRTCDataChannelRemoteHandlerConnection();
}

ScriptExecutionContextIdentifier SharedWorkerThreadProxy::loaderContextIdentifier() const
{
    return m_document->identifier();
}

void SharedWorkerThreadProxy::postTaskToLoader(ScriptExecutionContext::Task&& task)
{
    callOnMainThread([task = WTFMove(task), protectedThis = Ref { *this }] () mutable {
        task.performTask(protectedThis->m_document.get());
    });
}

bool SharedWorkerThreadProxy::postTaskForModeToWorkerOrWorkletGlobalScope(ScriptExecutionContext::Task&& task, const String& mode)
{
    if (m_isTerminatingOrTerminated)
        return false;

    m_workerThread->runLoop().postTaskForMode(WTFMove(task), mode);
    return true;
}

void SharedWorkerThreadProxy::postMessageToDebugger(const String&)
{

}

void SharedWorkerThreadProxy::setResourceCachingDisabledByWebInspector(bool)
{

}

void SharedWorkerThreadProxy::networkStateChanged(bool isOnLine)
{
    for (auto& proxy : allSharedWorkerThreadProxies().values())
        proxy->notifyNetworkStateChange(isOnLine);
}

void SharedWorkerThreadProxy::workerGlobalScopeClosed()
{
    callOnMainThread([identifier = thread().identifier()] {
        SharedWorkerContextManager::singleton().stopSharedWorker(identifier);
    });
}

ReportingClient* SharedWorkerThreadProxy::reportingClient() const
{
    return &m_document.get();
}

void SharedWorkerThreadProxy::setAppBadge(std::optional<uint64_t> badge)
{
    ASSERT(!isMainThread());
    callOnMainRunLoop([badge = WTFMove(badge), this, protectedThis = Ref { *this }] {
        m_page->badgeClient().setAppBadge(nullptr, m_clientOrigin.clientOrigin, badge);
    });
}

} // namespace WebCore
