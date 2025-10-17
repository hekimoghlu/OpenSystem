/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
#include "ServiceWorkerJob.h"

#include "HTTPHeaderNames.h"
#include "JSDOMPromiseDeferred.h"
#include "MIMETypeRegistry.h"
#include "ResourceError.h"
#include "ResourceResponse.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include "ServiceWorkerJobData.h"
#include "ServiceWorkerRegistration.h"
#include "WorkerFetchResult.h"
#include "WorkerRunLoop.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ServiceWorkerJob);

ServiceWorkerJob::ServiceWorkerJob(ServiceWorkerJobClient& client, RefPtr<DeferredPromise>&& promise, ServiceWorkerJobData&& jobData)
    : m_client(client)
    , m_jobData(WTFMove(jobData))
    , m_promise(WTFMove(promise))
    , m_contextIdentifier(client.contextIdentifier())
{
}

ServiceWorkerJob::~ServiceWorkerJob()
{
    ASSERT(m_creationThread.ptr() == &Thread::current());
}

RefPtr<DeferredPromise> ServiceWorkerJob::takePromise()
{
    return WTFMove(m_promise);
}

void ServiceWorkerJob::failedWithException(const Exception& exception)
{
    ASSERT(m_creationThread.ptr() == &Thread::current());
    ASSERT(!m_completed);

    m_completed = true;
    m_client.jobFailedWithException(*this, exception);
}

void ServiceWorkerJob::resolvedWithRegistration(ServiceWorkerRegistrationData&& data, ShouldNotifyWhenResolved shouldNotifyWhenResolved)
{
    ASSERT(m_creationThread.ptr() == &Thread::current());
    ASSERT(!m_completed);

    m_completed = true;
    m_client.jobResolvedWithRegistration(*this, WTFMove(data), shouldNotifyWhenResolved);
}

void ServiceWorkerJob::resolvedWithUnregistrationResult(bool unregistrationResult)
{
    ASSERT(m_creationThread.ptr() == &Thread::current());
    ASSERT(!m_completed);

    m_completed = true;
    m_client.jobResolvedWithUnregistrationResult(*this, unregistrationResult);
}

void ServiceWorkerJob::startScriptFetch(FetchOptions::Cache cachePolicy)
{
    ASSERT(m_creationThread.ptr() == &Thread::current());
    ASSERT(!m_completed);

    m_client.startScriptFetchForJob(*this, cachePolicy);
}

static ResourceRequest scriptResourceRequest(ScriptExecutionContext& context, const URL& url)
{
    ResourceRequest request { url };
    request.setInitiatorIdentifier(context.resourceRequestIdentifier());
    return request;
}

static FetchOptions scriptFetchOptions(FetchOptions::Cache cachePolicy, FetchOptions::Destination destination)
{
    FetchOptions options;
    options.mode = FetchOptions::Mode::SameOrigin;
    options.cache = cachePolicy;
    options.redirect = FetchOptions::Redirect::Error;
    options.destination = destination;
    options.credentials = FetchOptions::Credentials::SameOrigin;
    return options;
}

void ServiceWorkerJob::fetchScriptWithContext(ScriptExecutionContext& context, FetchOptions::Cache cachePolicy)
{
    ASSERT(m_creationThread.ptr() == &Thread::current());
    ASSERT(!m_completed);

    auto source = m_jobData.workerType == WorkerType::Module ? WorkerScriptLoader::Source::ModuleScript : WorkerScriptLoader::Source::ClassicWorkerScript;

    m_scriptLoader = WorkerScriptLoader::create();
    auto request = scriptResourceRequest(context, m_jobData.scriptURL);
    request.addHTTPHeaderField(HTTPHeaderName::ServiceWorker, "script"_s);

    m_scriptLoader->loadAsynchronously(context, WTFMove(request), source, scriptFetchOptions(cachePolicy, FetchOptions::Destination::Serviceworker), ContentSecurityPolicyEnforcement::DoNotEnforce, ServiceWorkersMode::None, *this, WorkerRunLoop::defaultMode());
}

ResourceError ServiceWorkerJob::validateServiceWorkerResponse(const ServiceWorkerJobData& jobData, const ResourceResponse& response)
{
    // Extract a MIME type from the response's header list. If this MIME type (ignoring parameters) is not a JavaScript MIME type, then:
    if (!MIMETypeRegistry::isSupportedJavaScriptMIMEType(response.mimeType()))
        return { errorDomainWebKitInternal, 0, response.url(), "MIME Type is not a JavaScript MIME type"_s };

    auto serviceWorkerAllowed = response.httpHeaderField(HTTPHeaderName::ServiceWorkerAllowed);
    String maxScopeString;
    if (serviceWorkerAllowed.isNull()) {
        auto path = jobData.scriptURL.path();
        // Last part of the path is the script's filename.
        maxScopeString = path.left(path.reverseFind('/') + 1).toString();
    } else {
        auto maxScope = URL(jobData.scriptURL, serviceWorkerAllowed);
        if (SecurityOrigin::create(maxScope)->isSameOriginAs(SecurityOrigin::create(jobData.scriptURL)))
            maxScopeString = maxScope.path().toString();
    }

    auto scopeString = jobData.scopeURL.path();
    if (maxScopeString.isNull() || !scopeString.startsWith(maxScopeString))
        return { errorDomainWebKitInternal, 0, response.url(), "Scope URL should start with the given script URL"_s };

    return { };
}

void ServiceWorkerJob::didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse& response)
{
    ASSERT(m_creationThread.ptr() == &Thread::current());
    ASSERT(!m_completed);
    ASSERT(m_scriptLoader);

    auto error = validateServiceWorkerResponse(m_jobData, response);
    if (error.isNull())
        return;

    m_scriptLoader->cancel();
    m_scriptLoader = nullptr;

    Exception exception { ExceptionCode::SecurityError, error.localizedDescription() };
    m_client.jobFailedLoadingScript(*this, WTFMove(error), WTFMove(exception));
}

void ServiceWorkerJob::notifyFinished(std::optional<ScriptExecutionContextIdentifier>)
{
    ASSERT(m_creationThread.ptr() == &Thread::current());
    ASSERT(m_scriptLoader);

    auto scriptLoader = std::exchange(m_scriptLoader, { });

    if (!scriptLoader->failed()) {
        m_client.jobFinishedLoadingScript(*this, scriptLoader->fetchResult());
        return;
    }

    auto& error = scriptLoader->error();
    ASSERT(!error.isNull());

    m_client.jobFailedLoadingScript(*this, error, Exception { error.isAccessControl() ? ExceptionCode::SecurityError : ExceptionCode::TypeError, makeString("Script "_s, scriptLoader->url().string(), " load failed"_s) });
}

bool ServiceWorkerJob::cancelPendingLoad()
{
    if (auto loader = std::exchange(m_scriptLoader, { })) {
        loader->cancel();
        return true;
    }
    return false;
}

} // namespace WebCore
