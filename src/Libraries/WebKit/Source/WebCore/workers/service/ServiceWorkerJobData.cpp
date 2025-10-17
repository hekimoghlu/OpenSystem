/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#include "ServiceWorkerJobData.h"

#include <wtf/CrossThreadCopier.h>

namespace WebCore {

ServiceWorkerJobData::ServiceWorkerJobData(SWServerConnectionIdentifier connectionIdentifier, const ServiceWorkerOrClientIdentifier& localSourceContext)
    : sourceContext(localSourceContext)
    , m_identifier { connectionIdentifier, ServiceWorkerJobIdentifier::generate() }
{
}

ServiceWorkerJobData::ServiceWorkerJobData(Identifier identifier, const ServiceWorkerOrClientIdentifier& localSourceContext)
    : sourceContext(localSourceContext)
    , m_identifier { identifier }
{
}

ServiceWorkerJobData::ServiceWorkerJobData(WebCore::ServiceWorkerJobDataIdentifier&& identifier, URL&& scriptURL, URL&& clientCreationURL, WebCore::SecurityOriginData&& topOrigin, URL&& scopeURL, WebCore::ServiceWorkerOrClientIdentifier&& sourceContext, WebCore::WorkerType workerType, WebCore::ServiceWorkerJobType type, String&& domainForCachePartition, bool isFromServiceWorkerPage, std::optional<WebCore::ServiceWorkerRegistrationOptions>&& registrationOptions)
    : scriptURL(WTFMove(scriptURL))
    , clientCreationURL(WTFMove(clientCreationURL))
    , topOrigin(WTFMove(topOrigin))
    , scopeURL(WTFMove(scopeURL))
    , sourceContext(WTFMove(sourceContext))
    , workerType(workerType)
    , type(type)
    , domainForCachePartition(WTFMove(domainForCachePartition))
    , isFromServiceWorkerPage(isFromServiceWorkerPage)
    , registrationOptions(WTFMove(registrationOptions))
    , m_identifier(WTFMove(identifier))
{
}

ServiceWorkerRegistrationKey ServiceWorkerJobData::registrationKey() const
{
    URL scope = scopeURL;
    scope.removeFragmentIdentifier();
    return { SecurityOriginData { topOrigin }, WTFMove(scope) };
}

std::optional<ScriptExecutionContextIdentifier> ServiceWorkerJobData::serviceWorkerPageIdentifier() const
{
    if (isFromServiceWorkerPage && std::holds_alternative<ScriptExecutionContextIdentifier>(sourceContext))
        return std::get<ScriptExecutionContextIdentifier>(sourceContext);
    return std::nullopt;
}

ServiceWorkerJobData ServiceWorkerJobData::isolatedCopy() const
{
    ServiceWorkerJobData result { identifier(), sourceContext };
    result.workerType = workerType;
    result.type = type;
    result.isFromServiceWorkerPage = isFromServiceWorkerPage;

    result.scriptURL = scriptURL.isolatedCopy();
    result.clientCreationURL = clientCreationURL.isolatedCopy();
    result.topOrigin = topOrigin.isolatedCopy();
    result.scopeURL = scopeURL.isolatedCopy();
    result.domainForCachePartition = domainForCachePartition.isolatedCopy();
    if (registrationOptions) {
        ASSERT(type == ServiceWorkerJobType::Register);
        result.registrationOptions = crossThreadCopy(registrationOptions);
    }
    return result;
}

// https://w3c.github.io/ServiceWorker/#dfn-job-equivalent
bool ServiceWorkerJobData::isEquivalent(const ServiceWorkerJobData& job) const
{
    if (type != job.type)
        return false;

    switch (type) {
    case ServiceWorkerJobType::Register:
        ASSERT(registrationOptions && job.registrationOptions);
        return scopeURL == job.scopeURL
            && scriptURL == job.scriptURL
            && workerType == job.workerType
            && registrationOptions->updateViaCache == job.registrationOptions->updateViaCache;
    case ServiceWorkerJobType::Update:
        return scopeURL == job.scopeURL
            && scriptURL == job.scriptURL
            && workerType == job.workerType;
    case ServiceWorkerJobType::Unregister:
        return scopeURL == job.scopeURL;
    }
    return false;
}

} // namespace WebCore
