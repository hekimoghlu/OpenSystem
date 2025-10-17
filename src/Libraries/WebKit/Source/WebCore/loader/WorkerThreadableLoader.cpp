/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#include "WorkerThreadableLoader.h"

#include "ContentSecurityPolicy.h"
#include "DedicatedWorkerGlobalScope.h"
#include "Document.h"
#include "DocumentThreadableLoader.h"
#include "InspectorInstrumentation.h"
#include "Performance.h"
#include "ResourceError.h"
#include "ResourceRequest.h"
#include "ResourceResponse.h"
#include "ResourceTiming.h"
#include "SecurityOrigin.h"
#include "ServiceWorker.h"
#include "ServiceWorkerGlobalScope.h"
#include "ThreadableLoader.h"
#include "WorkerGlobalScope.h"
#include "WorkerLoaderProxy.h"
#include "WorkerOrWorkletGlobalScope.h"
#include "WorkerThread.h"
#include <wtf/MainThread.h>
#include <wtf/Vector.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WorkerThreadableLoader::WorkerThreadableLoader(WorkerOrWorkletGlobalScope& workerOrWorkletGlobalScope, ThreadableLoaderClient& client, const String& taskMode, ResourceRequest&& request, const ThreadableLoaderOptions& options, const String& referrer)
    : m_workerClientWrapper(ThreadableLoaderClientWrapper::create(client, options.initiatorType))
    , m_bridge(*new MainThreadBridge(m_workerClientWrapper.get(), workerOrWorkletGlobalScope.workerOrWorkletThread()->workerLoaderProxy(), workerOrWorkletGlobalScope.identifier(), taskMode, WTFMove(request), options, referrer.isEmpty() ? workerOrWorkletGlobalScope.url().strippedForUseAsReferrer().string : referrer, workerOrWorkletGlobalScope))
{
}

WorkerThreadableLoader::~WorkerThreadableLoader()
{
    m_bridge.destroy();
}

void WorkerThreadableLoader::loadResourceSynchronously(WorkerOrWorkletGlobalScope& workerOrWorkletGlobalScope, ResourceRequest&& request, ThreadableLoaderClient& client, const ThreadableLoaderOptions& options)
{
    WorkerRunLoop& runLoop = workerOrWorkletGlobalScope.workerOrWorkletThread()->runLoop();

    // Create a unique mode just for this synchronous resource load.
    auto mode = makeString("loadResourceSynchronouslyMode"_s, runLoop.createUniqueId());

    Ref loader = WorkerThreadableLoader::create(workerOrWorkletGlobalScope, client, mode, WTFMove(request), options, String());
    bool success = true;
    while (!loader->done() && success)
        success = runLoop.runInMode(&workerOrWorkletGlobalScope, mode);

    if (!loader->done() && !success)
        loader->cancel();
}

void WorkerThreadableLoader::cancel()
{
    m_bridge.cancel();
}

void WorkerThreadableLoader::computeIsDone()
{
    m_bridge.computeIsDone();
}

struct LoaderTaskOptions {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    LoaderTaskOptions(const ThreadableLoaderOptions&, const String&, Ref<SecurityOrigin>&&);
    ThreadableLoaderOptions options;
    String referrer;
    Ref<SecurityOrigin> origin;
};

LoaderTaskOptions::LoaderTaskOptions(const ThreadableLoaderOptions& options, const String& referrer, Ref<SecurityOrigin>&& origin)
    : options(options.isolatedCopy())
    , referrer(referrer.isolatedCopy())
    , origin(WTFMove(origin))
{
}

WorkerThreadableLoader::MainThreadBridge::MainThreadBridge(ThreadableLoaderClientWrapper& workerClientWrapper, WorkerLoaderProxy* loaderProxy, ScriptExecutionContextIdentifier contextIdentifier, const String& taskMode,
    ResourceRequest&& request, const ThreadableLoaderOptions& options, const String& outgoingReferrer, WorkerOrWorkletGlobalScope& globalScope)
    : m_workerClientWrapper(&workerClientWrapper)
    , m_loaderProxy(loaderProxy)
    , m_taskMode(taskMode.isolatedCopy())
    , m_workerRequestIdentifier { ResourceLoaderIdentifier::generate() }
    , m_contextIdentifier(contextIdentifier)
{
    ASSERT(m_loaderProxy);
    RefPtr securityOrigin = globalScope.securityOrigin();
    CheckedPtr contentSecurityPolicy = globalScope.contentSecurityPolicy();

    ASSERT(securityOrigin);
    ASSERT(contentSecurityPolicy);

    Ref securityOriginCopy = securityOrigin->isolatedCopy();
    ReportingClient* reportingClient = nullptr;
    if (auto* client = m_loaderProxy ? m_loaderProxy->reportingClient() : nullptr)
        reportingClient = client;
    else if (auto* workerScope = dynamicDowncast<WorkerGlobalScope>(globalScope))
        reportingClient = workerScope;
    auto contentSecurityPolicyIsolatedCopy = makeUnique<ContentSecurityPolicy>(globalScope.url().isolatedCopy(), nullptr, reportingClient);
    contentSecurityPolicyIsolatedCopy->copyStateFrom(contentSecurityPolicy.get(), ContentSecurityPolicy::ShouldMakeIsolatedCopy::Yes);
    contentSecurityPolicyIsolatedCopy->copyUpgradeInsecureRequestStateFrom(*contentSecurityPolicy, ContentSecurityPolicy::ShouldMakeIsolatedCopy::Yes);
    auto crossOriginEmbedderPolicyCopy = globalScope.crossOriginEmbedderPolicy().isolatedCopy();

    auto optionsCopy = makeUnique<LoaderTaskOptions>(options, request.httpReferrer().isNull() ? outgoingReferrer : request.httpReferrer(), WTFMove(securityOriginCopy));

    // All loads start out as Document. Inside WorkerThreadableLoader we upgrade this to a Worker load.
    ASSERT(optionsCopy->options.initiatorContext == InitiatorContext::Document);
    optionsCopy->options.initiatorContext = InitiatorContext::Worker;

    if (optionsCopy->options.serviceWorkersMode == ServiceWorkersMode::All) {
        if (is<ServiceWorkerGlobalScope>(globalScope))
            optionsCopy->options.serviceWorkersMode = ServiceWorkersMode::None;
        else if (auto* activeServiceWorker = globalScope.activeServiceWorker()) {
            optionsCopy->options.serviceWorkerRegistrationIdentifier = activeServiceWorker->registrationIdentifier();
            optionsCopy->options.serviceWorkersMode = ServiceWorkersMode::All;
        } else if (is<DedicatedWorkerGlobalScope>(globalScope))
            optionsCopy->options.serviceWorkersMode = ServiceWorkersMode::None;
        else
            optionsCopy->options.serviceWorkersMode = ServiceWorkersMode::All;
    }
    if (!optionsCopy->options.clientIdentifier)
        optionsCopy->options.clientIdentifier = globalScope.identifier().object();

    if (RefPtr serviceWorkerGlobalScope = dynamicDowncast<ServiceWorkerGlobalScope>(globalScope))
        InspectorInstrumentation::willSendRequest(*serviceWorkerGlobalScope, m_workerRequestIdentifier, request);

    if (!m_loaderProxy)
        return;

    // Can we benefit from request being an r-value to create more efficiently its isolated copy?
    m_loaderProxy->postTaskToLoader([this, request = WTFMove(request).isolatedCopy(), options = WTFMove(optionsCopy), contentSecurityPolicyIsolatedCopy = WTFMove(contentSecurityPolicyIsolatedCopy), crossOriginEmbedderPolicyCopy = WTFMove(crossOriginEmbedderPolicyCopy)](ScriptExecutionContext& context) mutable {
        ASSERT(isMainThread());
        Ref document = downcast<Document>(context);

        // FIXME: If the site requests a local resource, then this will return a non-zero value but the sync path will return a 0 value.
        // Either this should return 0 or the other code path should call a failure callback.
        m_mainThreadLoader = DocumentThreadableLoader::create(document, *this, WTFMove(request), options->options, WTFMove(options->origin), WTFMove(contentSecurityPolicyIsolatedCopy), WTFMove(crossOriginEmbedderPolicyCopy), WTFMove(options->referrer), DocumentThreadableLoader::ShouldLogError::No);
        ASSERT(m_mainThreadLoader || m_loadingFinished);
    });
}

WorkerThreadableLoader::MainThreadBridge::~MainThreadBridge() = default;

void WorkerThreadableLoader::MainThreadBridge::destroy()
{
    // Ensure that no more client callbacks are done in the worker context's thread.
    clearClientWrapper();

    if (!m_loaderProxy)
        return;

    // "delete this" and m_mainThreadLoader::deref() on the worker object's thread.
    m_loaderProxy->postTaskToLoader([self = std::unique_ptr<WorkerThreadableLoader::MainThreadBridge>(this)] (ScriptExecutionContext& context) {
        ASSERT(isMainThread());
        ASSERT_UNUSED(context, context.isDocument());
        if (RefPtr mainThreadLoader = self->m_mainThreadLoader)
            mainThreadLoader->clearClient();
    });
}

void WorkerThreadableLoader::MainThreadBridge::cancel()
{
    if (m_loaderProxy) {
        m_loaderProxy->postTaskToLoader([this] (ScriptExecutionContext& context) {
            ASSERT(isMainThread());
            ASSERT_UNUSED(context, context.isDocument());

            if (RefPtr mainThreadLoader = std::exchange(m_mainThreadLoader, nullptr))
                mainThreadLoader->cancel();
        });
    }

    // Taking a ref of client wrapper as call to didFail may take out the last reference of it.
    Ref workerClientWrapper { *m_workerClientWrapper };
    // If the client hasn't reached a termination state, then transition it by sending a cancellation error.
    // Note: no more client callbacks will be done after this method -- we clear the client wrapper to ensure that.
    ResourceError error(ResourceError::Type::Cancellation);
    // FIXME: Always get the `ScriptExecutionContextIdentifier` of the `Document`.
    workerClientWrapper->didFail(std::nullopt, error);
    workerClientWrapper->clearClient();
}

void WorkerThreadableLoader::MainThreadBridge::computeIsDone()
{
    if (!m_loaderProxy)
        return;

    m_loaderProxy->postTaskToLoader([this](auto&) {
        if (RefPtr mainThreadLoader = m_mainThreadLoader) {
            mainThreadLoader->computeIsDone();
            return;
        }
        notifyIsDone(true);
    });
}

void WorkerThreadableLoader::MainThreadBridge::notifyIsDone(bool isDone)
{
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_contextIdentifier, [protectedWorkerClientWrapper = Ref { *m_workerClientWrapper }, isDone](auto&) {
        protectedWorkerClientWrapper->notifyIsDone(isDone);
    }, m_taskMode);
}

void WorkerThreadableLoader::MainThreadBridge::clearClientWrapper()
{
    protectedWorkerClientWrapper()->clearClient();
}

RefPtr<ThreadableLoaderClientWrapper> WorkerThreadableLoader::MainThreadBridge::protectedWorkerClientWrapper() const
{
    return m_workerClientWrapper;
}

void WorkerThreadableLoader::MainThreadBridge::didSendData(unsigned long long bytesSent, unsigned long long totalBytesToBeSent)
{
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_contextIdentifier, [protectedWorkerClientWrapper = Ref { *m_workerClientWrapper }, bytesSent, totalBytesToBeSent] (ScriptExecutionContext& context) mutable {
        ASSERT_UNUSED(context, context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        protectedWorkerClientWrapper->didSendData(bytesSent, totalBytesToBeSent);
    }, m_taskMode);
}

void WorkerThreadableLoader::MainThreadBridge::didReceiveResponse(ScriptExecutionContextIdentifier mainContextIdentifier, std::optional<ResourceLoaderIdentifier> identifier, const ResourceResponse& response)
{
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_contextIdentifier, [protectedWorkerClientWrapper = Ref { *m_workerClientWrapper }, workerRequestIdentifier = m_workerRequestIdentifier, mainContextIdentifier, identifier, responseData = response.crossThreadData()] (ScriptExecutionContext& context) mutable {
        ASSERT(context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        auto response = ResourceResponse::fromCrossThreadData(WTFMove(responseData));
        protectedWorkerClientWrapper->didReceiveResponse(mainContextIdentifier, identifier, response);
        if (auto* serviceWorkerGlobalScope = dynamicDowncast<ServiceWorkerGlobalScope>(context))
            InspectorInstrumentation::didReceiveResourceResponse(*serviceWorkerGlobalScope, workerRequestIdentifier, response);
    }, m_taskMode);
}

void WorkerThreadableLoader::MainThreadBridge::didReceiveData(const SharedBuffer& data)
{
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_contextIdentifier, [protectedWorkerClientWrapper = Ref { *m_workerClientWrapper }, workerRequestIdentifier = m_workerRequestIdentifier, buffer = Ref { data }] (ScriptExecutionContext& context) {
        ASSERT(context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        protectedWorkerClientWrapper->didReceiveData(buffer);
        if (auto* serviceWorkerGlobalScope = dynamicDowncast<ServiceWorkerGlobalScope>(context))
            InspectorInstrumentation::didReceiveData(*serviceWorkerGlobalScope, workerRequestIdentifier, buffer);
    }, m_taskMode);
}

void WorkerThreadableLoader::MainThreadBridge::didFinishLoading(ScriptExecutionContextIdentifier mainContext, std::optional<ResourceLoaderIdentifier> identifier, const NetworkLoadMetrics& metrics)
{
    m_loadingFinished = true;
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_contextIdentifier, [protectedWorkerClientWrapper = Ref { *m_workerClientWrapper }, workerRequestIdentifier = m_workerRequestIdentifier, metrics = metrics.isolatedCopy(), mainContext, identifier] (ScriptExecutionContext& context) mutable {
        ASSERT(context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        protectedWorkerClientWrapper->didFinishLoading(mainContext, identifier, metrics);
        if (auto* serviceWorkerGlobalScope = dynamicDowncast<ServiceWorkerGlobalScope>(context))
            InspectorInstrumentation::didFinishLoading(*serviceWorkerGlobalScope, workerRequestIdentifier, metrics);
    }, m_taskMode);
}

void WorkerThreadableLoader::MainThreadBridge::didFail(std::optional<ScriptExecutionContextIdentifier> mainContext, const ResourceError& error)
{
    m_loadingFinished = true;
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_contextIdentifier, [protectedWorkerClientWrapper = Ref { *m_workerClientWrapper }, workerRequestIdentifier = m_workerRequestIdentifier, mainContext, error = error.isolatedCopy()] (ScriptExecutionContext& context) mutable {
        ASSERT(context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        ThreadableLoader::logError(context, error, protectedWorkerClientWrapper->initiator());
        protectedWorkerClientWrapper->didFail(mainContext, error);
        if (auto* serviceWorkerGlobalScope = dynamicDowncast<ServiceWorkerGlobalScope>(context))
            InspectorInstrumentation::didFailLoading(*serviceWorkerGlobalScope, workerRequestIdentifier, error);
    }, m_taskMode);
}

void WorkerThreadableLoader::MainThreadBridge::didFinishTiming(const ResourceTiming& resourceTiming)
{
    m_networkLoadMetrics = resourceTiming.networkLoadMetrics();
    ScriptExecutionContext::postTaskForModeToWorkerOrWorklet(m_contextIdentifier, [protectedWorkerClientWrapper = Ref { *m_workerClientWrapper }, resourceTiming = resourceTiming.isolatedCopy()] (ScriptExecutionContext& context) mutable {
        ASSERT(context.isWorkerGlobalScope() || context.isWorkletGlobalScope());
        ASSERT(!resourceTiming.initiatorType().isEmpty());

        // No need to notify clients, just add the performance timing entry.
        if (auto* globalScope = dynamicDowncast<WorkerGlobalScope>(context))
            globalScope->protectedPerformance()->addResourceTiming(WTFMove(resourceTiming));
    }, m_taskMode);
}

} // namespace WebCore
