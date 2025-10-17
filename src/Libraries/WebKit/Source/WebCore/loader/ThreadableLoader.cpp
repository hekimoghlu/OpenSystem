/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 21, 2024.
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
#include "ThreadableLoader.h"

#include "CachedResourceRequestInitiatorTypes.h"
#include "Document.h"
#include "DocumentLoader.h"
#include "DocumentThreadableLoader.h"
#include "ResourceError.h"
#include "ScriptExecutionContext.h"
#include "WorkerGlobalScope.h"
#include "WorkerRunLoop.h"
#include "WorkerThreadableLoader.h"
#include "WorkletGlobalScope.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

ThreadableLoaderOptions::ThreadableLoaderOptions()
{
    mode = FetchOptions::Mode::SameOrigin;
}

ThreadableLoaderOptions::~ThreadableLoaderOptions() = default;

ThreadableLoaderOptions::ThreadableLoaderOptions(FetchOptions&& baseOptions)
    : ResourceLoaderOptions { WTFMove(baseOptions) }
{
}

ThreadableLoaderOptions::ThreadableLoaderOptions(const ResourceLoaderOptions& baseOptions, ContentSecurityPolicyEnforcement contentSecurityPolicyEnforcement, String&& initiatorType, ResponseFilteringPolicy filteringPolicy)
    : ResourceLoaderOptions(baseOptions)
    , contentSecurityPolicyEnforcement(contentSecurityPolicyEnforcement)
    , initiatorType(WTFMove(initiatorType))
    , filteringPolicy(filteringPolicy)
{
}

ThreadableLoaderOptions ThreadableLoaderOptions::isolatedCopy() const
{
    ThreadableLoaderOptions copy;

    // FetchOptions
    copy.destination = this->destination;
    copy.mode = this->mode;
    copy.credentials = this->credentials;
    copy.cache = this->cache;
    copy.redirect = this->redirect;
    copy.referrerPolicy = this->referrerPolicy;
    copy.integrity = this->integrity.isolatedCopy();
    copy.keepAlive = this->keepAlive;
    copy.clientIdentifier = this->clientIdentifier;
    copy.resultingClientIdentifier = this->resultingClientIdentifier;

    // ResourceLoaderOptions
    copy.sendLoadCallbacks = this->sendLoadCallbacks;
    copy.sniffContent = this->sniffContent;
    copy.contentEncodingSniffingPolicy = this->contentEncodingSniffingPolicy;
    copy.dataBufferingPolicy = this->dataBufferingPolicy;
    copy.storedCredentialsPolicy = this->storedCredentialsPolicy;
    copy.securityCheck = this->securityCheck;
    copy.certificateInfoPolicy = this->certificateInfoPolicy;
    copy.contentSecurityPolicyImposition = this->contentSecurityPolicyImposition;
    copy.defersLoadingPolicy = this->defersLoadingPolicy;
    copy.cachingPolicy = this->cachingPolicy;
    copy.sameOriginDataURLFlag = this->sameOriginDataURLFlag;
    copy.initiatorContext = this->initiatorContext;
    copy.initiator = this->initiator;
    copy.clientCredentialPolicy = this->clientCredentialPolicy;
    copy.maxRedirectCount = this->maxRedirectCount;
    copy.preflightPolicy = this->preflightPolicy;
    copy.navigationPreloadIdentifier = this->navigationPreloadIdentifier;
    copy.fetchPriority = this->fetchPriority;
    copy.shouldEnableContentExtensionsCheck = this->shouldEnableContentExtensionsCheck;

    // ThreadableLoaderOptions
    copy.contentSecurityPolicyEnforcement = this->contentSecurityPolicyEnforcement;
    copy.initiatorType = this->initiatorType.isolatedCopy();
    copy.filteringPolicy = this->filteringPolicy;

    return copy;
}


RefPtr<ThreadableLoader> ThreadableLoader::create(ScriptExecutionContext& context, ThreadableLoaderClient& client, ResourceRequest&& request, const ThreadableLoaderOptions& options, String&& referrer, String&& taskMode)
{
    RefPtr document = [&] {
        if (auto* globalScope = dynamicDowncast<WorkletGlobalScope>(context))
            return globalScope->responsibleDocument();
        return dynamicDowncast<Document>(context);
    }();

    if (auto* documentLoader = document ? document->loader() : nullptr)
        request.setIsAppInitiated(documentLoader->lastNavigationWasAppInitiated());
    
    if (is<WorkerGlobalScope>(context) || (is<WorkletGlobalScope>(context) && downcast<WorkletGlobalScope>(context).workerOrWorkletThread()))
        return WorkerThreadableLoader::create(static_cast<WorkerOrWorkletGlobalScope&>(context), client, WTFMove(taskMode), WTFMove(request), options, WTFMove(referrer));

    return DocumentThreadableLoader::create(*document, client, WTFMove(request), options, WTFMove(referrer));
}

void ThreadableLoader::loadResourceSynchronously(ScriptExecutionContext& context, ResourceRequest&& request, ThreadableLoaderClient& client, const ThreadableLoaderOptions& options)
{
    auto resourceURL = request.url();
    if (auto* globalScope = dynamicDowncast<WorkerGlobalScope>(context))
        WorkerThreadableLoader::loadResourceSynchronously(*globalScope, WTFMove(request), client, options);
    else
        DocumentThreadableLoader::loadResourceSynchronously(downcast<Document>(context), WTFMove(request), client, options);
    context.didLoadResourceSynchronously(resourceURL);
}

void ThreadableLoader::logError(ScriptExecutionContext& context, const ResourceError& error, const String& initiatorType)
{
    if (error.isCancellation())
        return;

    // FIXME: Some errors are returned with null URLs. This leads to poor console messages. We should do better for these errors.
    if (error.failingURL().isNull())
        return;

    // We further reduce logging to some errors.
    // FIXME: Log more errors when making so do not make some layout tests flaky.
    if (error.domain() != errorDomainWebKitInternal && error.domain() != errorDomainWebKitServiceWorker && !error.isAccessControl())
        return;

    ASCIILiteral messageStart;
    if (initiatorType == cachedResourceRequestInitiatorTypes().eventsource)
        messageStart = "EventSource cannot load "_s;
    else if (initiatorType == cachedResourceRequestInitiatorTypes().fetch)
        messageStart = "Fetch API cannot load "_s;
    else if (initiatorType == cachedResourceRequestInitiatorTypes().xmlhttprequest)
        messageStart = "XMLHttpRequest cannot load "_s;
    else
        messageStart = "Cannot load "_s;

    String messageEnd = error.isAccessControl() ? " due to access control checks."_s : "."_s;
    context.addConsoleMessage(MessageSource::JS, MessageLevel::Error, makeString(messageStart, error.failingURL().string(), messageEnd));
}

} // namespace WebCore
