/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#include "ServiceWorkerGlobalScope.h"

#include "Document.h"
#include "EventLoop.h"
#include "EventNames.h"
#include "ExtendableEvent.h"
#include "FetchEvent.h"
#include "FrameLoader.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalFrame.h"
#include "LocalFrameLoaderClient.h"
#include "Logging.h"
#include "NotificationEvent.h"
#include "PushEvent.h"
#include "SWContextManager.h"
#include "SWServer.h"
#include "ServiceWorker.h"
#include "ServiceWorkerClient.h"
#include "ServiceWorkerClients.h"
#include "ServiceWorkerContainer.h"
#include "ServiceWorkerThread.h"
#include "ServiceWorkerWindowClient.h"
#include "WebCoreJSClientData.h"
#include "WorkerNavigator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ServiceWorkerGlobalScope);

Ref<ServiceWorkerGlobalScope> ServiceWorkerGlobalScope::create(ServiceWorkerContextData&& contextData, ServiceWorkerData&& workerData, const WorkerParameters& params, Ref<SecurityOrigin>&& origin, ServiceWorkerThread& thread, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy* connectionProxy, SocketProvider* socketProvider, std::unique_ptr<NotificationClient>&& notificationClient, std::unique_ptr<WorkerClient>&& workerClient)
{
    auto scope = adoptRef(*new ServiceWorkerGlobalScope { WTFMove(contextData), WTFMove(workerData), params, WTFMove(origin), thread, WTFMove(topOrigin), connectionProxy, socketProvider, WTFMove(notificationClient), WTFMove(workerClient) });
    scope->addToContextsMap();
    scope->applyContentSecurityPolicyResponseHeaders(params.contentSecurityPolicyResponseHeaders);
    scope->notifyServiceWorkerPageOfCreationIfNecessary();
    return scope;
}

ServiceWorkerGlobalScope::ServiceWorkerGlobalScope(ServiceWorkerContextData&& contextData, ServiceWorkerData&& workerData, const WorkerParameters& params, Ref<SecurityOrigin>&& origin, ServiceWorkerThread& thread, Ref<SecurityOrigin>&& topOrigin, IDBClient::IDBConnectionProxy* connectionProxy, SocketProvider* socketProvider, std::unique_ptr<NotificationClient>&& notificationClient, std::unique_ptr<WorkerClient>&& workerClient)
    : WorkerGlobalScope(WorkerThreadType::ServiceWorker, params, WTFMove(origin), thread, WTFMove(topOrigin), connectionProxy, socketProvider, WTFMove(workerClient))
    , m_contextData(WTFMove(contextData))
    , m_registration(ServiceWorkerRegistration::getOrCreate(*this, navigator().serviceWorker(), WTFMove(m_contextData.registration)))
    , m_serviceWorker(ServiceWorker::getOrCreate(*this, WTFMove(workerData)))
    , m_clients(ServiceWorkerClients::create())
    , m_notificationClient(WTFMove(notificationClient))
    , m_userGestureTimer(*this, &ServiceWorkerGlobalScope::resetUserGesture)
{
}

ServiceWorkerGlobalScope::~ServiceWorkerGlobalScope()
{
    // We need to remove from the contexts map very early in the destructor so that calling postTask() on this WorkerGlobalScope from another thread is safe.
    removeFromContextsMap();

    // NotificationClient might have some interactions pending with the main thread,
    // so it should also be destroyed there.
    callOnMainThread([notificationClient = WTFMove(m_notificationClient)] { });
}

void ServiceWorkerGlobalScope::dispatchPushEvent(PushEvent& pushEvent)
{
#if ENABLE(DECLARATIVE_WEB_PUSH)
    ASSERT(!m_declarativePushEvent && !m_pushEvent);
#else
    ASSERT(!m_pushEvent);
#endif

    m_pushEvent = &pushEvent;
    m_lastPushEventTime = MonotonicTime::now();
    dispatchEvent(pushEvent);
    m_pushEvent = nullptr;
}

#if ENABLE(DECLARATIVE_WEB_PUSH)
void ServiceWorkerGlobalScope::dispatchDeclarativePushEvent(PushEvent& event)
{
    ASSERT(!m_declarativePushEvent && !m_pushEvent);
    m_declarativePushEvent = &event;
    m_lastPushEventTime = MonotonicTime::now();
    dispatchEvent(event);
}

void ServiceWorkerGlobalScope::clearDeclarativePushEvent()
{
    ASSERT(m_declarativePushEvent);
    m_declarativePushEvent = nullptr;
}
#endif

void ServiceWorkerGlobalScope::notifyServiceWorkerPageOfCreationIfNecessary()
{
    auto serviceWorkerPage = this->serviceWorkerPage();
    if (!serviceWorkerPage)
        return;

    ASSERT(isMainThread());
    serviceWorkerPage->setServiceWorkerGlobalScope(*this);

    if (auto* localMainFrame = dynamicDowncast<LocalFrame>(serviceWorkerPage->mainFrame())) {
        // FIXME: We currently do not support non-normal worlds in service workers.
        Ref normalWorld = static_cast<JSVMClientData*>(vm().clientData)->normalWorld();
        localMainFrame->loader().client().dispatchServiceWorkerGlobalObjectAvailable(normalWorld);
    }
}

Page* ServiceWorkerGlobalScope::serviceWorkerPage()
{
    if (!m_contextData.serviceWorkerPageIdentifier)
        return nullptr;

    RELEASE_ASSERT(isMainThread());
    return Page::serviceWorkerPage(*m_contextData.serviceWorkerPageIdentifier);
}

void ServiceWorkerGlobalScope::skipWaiting(Ref<DeferredPromise>&& promise)
{
    RELEASE_LOG(ServiceWorker, "ServiceWorkerGlobalScope::skipWaiting for worker %" PRIu64, thread().identifier().toUInt64());

    uint64_t requestIdentifier = ++m_lastRequestIdentifier;
    m_pendingSkipWaitingPromises.add(requestIdentifier, WTFMove(promise));

    callOnMainThread([workerThread = Ref { thread() }, requestIdentifier]() mutable {
        if (auto* connection = SWContextManager::singleton().connection()) {
            auto identifier = workerThread->identifier();
            connection->skipWaiting(identifier, [workerThread = WTFMove(workerThread), requestIdentifier] {
                workerThread->runLoop().postTask([requestIdentifier](auto& context) {
                    auto& scope = downcast<ServiceWorkerGlobalScope>(context);
                    scope.eventLoop().queueTask(TaskSource::DOMManipulation, [scope = Ref { scope }, requestIdentifier]() mutable {
                        if (auto promise = scope->m_pendingSkipWaitingPromises.take(requestIdentifier))
                            promise->resolve();
                    });
                });
            });
        }
    });
}

enum EventTargetInterfaceType ServiceWorkerGlobalScope::eventTargetInterface() const
{
    return EventTargetInterfaceType::ServiceWorkerGlobalScope;
}

ServiceWorkerThread& ServiceWorkerGlobalScope::thread()
{
    return static_cast<ServiceWorkerThread&>(WorkerGlobalScope::thread());
}

void ServiceWorkerGlobalScope::prepareForDestruction()
{
    // Make sure we destroy fetch events objects before the VM goes away, since their
    // destructor may access the VM.
    m_extendedEvents.clear();

    auto ongoingFetchTasks = std::exchange(m_ongoingFetchTasks, { });
    for (auto& task : ongoingFetchTasks.values())
        task.client->contextIsStopping();

    WorkerGlobalScope::prepareForDestruction();
}

// https://w3c.github.io/ServiceWorker/#update-service-worker-extended-events-set-algorithm
void ServiceWorkerGlobalScope::updateExtendedEventsSet(ExtendableEvent* newEvent)
{
    ASSERT(isContextThread());
    ASSERT(!newEvent || !newEvent->isBeingDispatched());
    bool hadPendingEvents = hasPendingEvents();
    m_extendedEvents.removeAllMatching([](auto& event) {
        return !event->pendingPromiseCount();
    });

    if (newEvent && newEvent->pendingPromiseCount()) {
        m_extendedEvents.append(*newEvent);
        newEvent->whenAllExtendLifetimePromisesAreSettled([this](auto&&) {
            this->updateExtendedEventsSet();
        });
        // Clear out the event's target as it is the WorkerGlobalScope and we do not want to keep it
        // alive unnecessarily.
        newEvent->setTarget(nullptr);
    }

    bool hasPendingEvents = this->hasPendingEvents();
    if (hasPendingEvents == hadPendingEvents)
        return;

    callOnMainThread([threadIdentifier = thread().identifier(), hasPendingEvents] {
        if (auto* connection = SWContextManager::singleton().connection())
            connection->setServiceWorkerHasPendingEvents(threadIdentifier, hasPendingEvents);
    });
}

const ServiceWorkerContextData::ImportedScript* ServiceWorkerGlobalScope::scriptResource(const URL& url) const
{
    auto iterator = m_contextData.scriptResourceMap.find(url);
    return iterator == m_contextData.scriptResourceMap.end() ? nullptr : &iterator->value;
}

void ServiceWorkerGlobalScope::setScriptResource(const URL& url, ServiceWorkerContextData::ImportedScript&& script)
{
    callOnMainThread([threadIdentifier = thread().identifier(), url = url.isolatedCopy(), script = script.isolatedCopy()] {
        if (auto* connection = SWContextManager::singleton().connection())
            connection->setScriptResource(threadIdentifier, url, script);
    });

    m_contextData.scriptResourceMap.set(url, WTFMove(script));
}

void ServiceWorkerGlobalScope::didSaveScriptsToDisk(ScriptBuffer&& script, HashMap<URL, ScriptBuffer>&& importedScripts)
{
    // These scripts should be identical to the ones we have. However, these are mmap'd so using them helps reduce dirty memory usage.
    updateSourceProviderBuffers(script, importedScripts);

    if (script) {
        ASSERT(m_contextData.script == script);
        m_contextData.script = WTFMove(script);
    }
    for (auto& pair : importedScripts) {
        auto it = m_contextData.scriptResourceMap.find(pair.key);
        if (it == m_contextData.scriptResourceMap.end())
            continue;
        ASSERT(it->value.script == pair.value); // Do a memcmp to make sure the scripts are identical.
        it->value.script = WTFMove(pair.value);
    }
}

void ServiceWorkerGlobalScope::recordUserGesture()
{
    m_isProcessingUserGesture = true;
    m_userGestureTimer.startOneShot(userGestureLifetime);
}

bool ServiceWorkerGlobalScope::didFirePushEventRecently() const
{
    return MonotonicTime::now() <= m_lastPushEventTime + SWServer::defaultTerminationDelay;
}

void ServiceWorkerGlobalScope::addConsoleMessage(MessageSource source, MessageLevel level, const String& message, unsigned long requestIdentifier)
{
    if (m_consoleMessageReportingEnabled) {
        callOnMainThread([threadIdentifier = thread().identifier(), source, level, message = message.isolatedCopy(), requestIdentifier] {
            if (auto* connection = SWContextManager::singleton().connection())
                connection->reportConsoleMessage(threadIdentifier, source, level, message, requestIdentifier);
        });
    }
    WorkerGlobalScope::addConsoleMessage(source, level, message, requestIdentifier);
}

CookieStore& ServiceWorkerGlobalScope::cookieStore()
{
    if (!m_cookieStore)
        m_cookieStore = CookieStore::create(this);
    return *m_cookieStore;
}

void ServiceWorkerGlobalScope::addFetchTask(FetchKey key, Ref<ServiceWorkerFetch::Client>&& client)
{
    ASSERT(!m_ongoingFetchTasks.contains(key));
    m_ongoingFetchTasks.add(key, FetchTask { WTFMove(client), nullptr });
}

void ServiceWorkerGlobalScope::addFetchEvent(FetchKey key, FetchEvent& event)
{
    ASSERT(m_ongoingFetchTasks.contains(key));
    auto iterator = m_ongoingFetchTasks.find(key);

    bool isHandled = WTF::switchOn(iterator->value.navigationPreload, [] (std::nullptr_t) {
        return false;
    }, [] (Ref<FetchEvent>&) {
        ASSERT_NOT_REACHED();
        return false;
    }, [&event] (UniqueRef<ResourceResponse>& response) {
        event.navigationPreloadIsReady(WTFMove(response.get()));
        return true;
    }, [&event] (UniqueRef<ResourceError>& error) {
        event.navigationPreloadFailed(WTFMove(error.get()));
        return true;
    });

    if (isHandled)
        iterator->value.navigationPreload = nullptr;
    else
        iterator->value.navigationPreload = Ref { event };
}

void ServiceWorkerGlobalScope::removeFetchTask(FetchKey key)
{
    m_ongoingFetchTasks.remove(key);
}

RefPtr<ServiceWorkerFetch::Client> ServiceWorkerGlobalScope::fetchTask(FetchKey key)
{
    auto iterator = m_ongoingFetchTasks.find(key);
    return iterator != m_ongoingFetchTasks.end() ? iterator->value.client.get() : nullptr;
}

RefPtr<ServiceWorkerFetch::Client> ServiceWorkerGlobalScope::takeFetchTask(FetchKey key)
{
    return m_ongoingFetchTasks.take(key).client;
}

bool ServiceWorkerGlobalScope::hasFetchTask() const
{
    return !m_ongoingFetchTasks.isEmpty();
}

void ServiceWorkerGlobalScope::navigationPreloadFailed(FetchKey key, ResourceError&& error)
{
    auto iterator = m_ongoingFetchTasks.find(key);
    if (iterator == m_ongoingFetchTasks.end())
        return;

    if (std::holds_alternative<Ref<FetchEvent>>(iterator->value.navigationPreload)) {
        std::get<Ref<FetchEvent>>(iterator->value.navigationPreload)->navigationPreloadFailed(WTFMove(error));
        iterator->value.navigationPreload = nullptr;
        return;
    }

    iterator->value.navigationPreload = makeUniqueRef<ResourceError>(WTFMove(error));
}

void ServiceWorkerGlobalScope::navigationPreloadIsReady(FetchKey key, ResourceResponse&& response)
{
    auto iterator = m_ongoingFetchTasks.find(key);
    if (iterator == m_ongoingFetchTasks.end())
        return;

    if (std::holds_alternative<Ref<FetchEvent>>(iterator->value.navigationPreload)) {
        std::get<Ref<FetchEvent>>(iterator->value.navigationPreload)->navigationPreloadIsReady(WTFMove(response));
        iterator->value.navigationPreload = nullptr;
        return;
    }

    iterator->value.navigationPreload = makeUniqueRef<ResourceResponse>(WTFMove(response));
}

} // namespace WebCore
