/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#include "FetchLoader.h"

#include "BlobURL.h"
#include "CachedResourceRequestInitiatorTypes.h"
#include "ContentSecurityPolicy.h"
#include "FetchBody.h"
#include "FetchBodyConsumer.h"
#include "FetchLoaderClient.h"
#include "FetchRequest.h"
#include "ResourceError.h"
#include "ResourceRequest.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include "SharedBuffer.h"
#include "TextResourceDecoder.h"
#include "ThreadableBlobRegistry.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FetchLoaderClient);

void FetchLoader::start(ScriptExecutionContext& context, const Blob& blob)
{
    return startLoadingBlobURL(context, blob.url());
}

void FetchLoader::startLoadingBlobURL(ScriptExecutionContext& context, const URL& blobURL)
{
    m_urlForReading = { BlobURL::createPublicURL(context.securityOrigin()), context.topOrigin().data() };

    if (m_urlForReading.isEmpty()) {
        m_client->didFail({ errorDomainWebKitInternal, 0, URL(), "Could not create URL for Blob"_s });
        return;
    }

    ThreadableBlobRegistry::registerBlobURL(context.securityOrigin(), context.policyContainer(), m_urlForReading, blobURL);

    ResourceRequest request(m_urlForReading);
    request.setInitiatorIdentifier(context.resourceRequestIdentifier());
    request.setHTTPMethod("GET"_s);

    ThreadableLoaderOptions options;
    options.sendLoadCallbacks = SendCallbackPolicy::SendCallbacks;
    options.dataBufferingPolicy = DataBufferingPolicy::DoNotBufferData;
    options.preflightPolicy = PreflightPolicy::Consider;
    options.credentials = FetchOptions::Credentials::Include;
    options.mode = FetchOptions::Mode::SameOrigin;
    options.contentSecurityPolicyEnforcement = ContentSecurityPolicyEnforcement::DoNotEnforce;

    m_loader = ThreadableLoader::create(context, *this, WTFMove(request), options);
    m_isStarted = m_loader;
}

void FetchLoader::start(ScriptExecutionContext& context, const FetchRequest& request, const String& initiator)
{
    ResourceLoaderOptions resourceLoaderOptions = request.fetchOptions();
    resourceLoaderOptions.preflightPolicy = PreflightPolicy::Consider;
    ThreadableLoaderOptions options(resourceLoaderOptions,
        context.shouldBypassMainWorldContentSecurityPolicy() ? ContentSecurityPolicyEnforcement::DoNotEnforce : ContentSecurityPolicyEnforcement::EnforceConnectSrcDirective,
        String(initiator),
        ResponseFilteringPolicy::Disable);
    options.sendLoadCallbacks = SendCallbackPolicy::SendCallbacks;
    options.dataBufferingPolicy = DataBufferingPolicy::DoNotBufferData;
    options.sameOriginDataURLFlag = SameOriginDataURLFlag::Set;
    options.navigationPreloadIdentifier = request.navigationPreloadIdentifier();
    options.contentEncodingSniffingPolicy = ContentEncodingSniffingPolicy::Disable;
    options.fetchPriority = request.priority();
    options.shouldEnableContentExtensionsCheck = request.shouldEnableContentExtensionsCheck() ? ShouldEnableContentExtensionsCheck::Yes : ShouldEnableContentExtensionsCheck::No;

    ResourceRequest fetchRequest = request.resourceRequest();

    ASSERT(context.contentSecurityPolicy());
    {
        CheckedRef contentSecurityPolicy = *context.contentSecurityPolicy();

        contentSecurityPolicy->upgradeInsecureRequestIfNeeded(fetchRequest, ContentSecurityPolicy::InsecureRequestType::Load);

        if (!context.shouldBypassMainWorldContentSecurityPolicy() && !contentSecurityPolicy->allowConnectToSource(fetchRequest.url())) {
            m_client->didFail({ errorDomainWebKitInternal, 0, fetchRequest.url(), "Not allowed by ContentSecurityPolicy"_s, ResourceError::Type::AccessControl });
            return;
        }
    }

    String referrer = request.internalRequestReferrer();
    if (referrer == "no-referrer"_s) {
        options.referrerPolicy = ReferrerPolicy::NoReferrer;
        referrer = String();
    } else
        referrer = (referrer == "client"_s) ? context.url().strippedForUseAsReferrer().string : URL(context.url(), referrer).strippedForUseAsReferrer().string;
    if (options.referrerPolicy == ReferrerPolicy::EmptyString)
        options.referrerPolicy = context.referrerPolicy();

    m_loader = ThreadableLoader::create(context, *this, WTFMove(fetchRequest), options, WTFMove(referrer));
    m_isStarted = m_loader;
}

FetchLoader::FetchLoader(FetchLoaderClient& client, FetchBodyConsumer* consumer)
    : m_client(client)
    , m_consumer(consumer)
{
}

FetchLoader::~FetchLoader() = default;

void FetchLoader::stop()
{
    if (m_consumer)
        m_consumer->clean();
    if (m_loader)
        m_loader->cancel();
}

RefPtr<FragmentedSharedBuffer> FetchLoader::startStreaming()
{
    ASSERT(m_consumer);
    auto firstChunk = m_consumer->takeData();
    m_consumer = nullptr;
    return firstChunk;
}

void FetchLoader::didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse& response)
{
    m_client->didReceiveResponse(response);
}

void FetchLoader::didReceiveData(const SharedBuffer& buffer)
{
    if (!m_consumer) {
        m_client->didReceiveData(buffer);
        return;
    }
    m_consumer->append(buffer);
}

void FetchLoader::didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics& metrics)
{
    m_client->didSucceed(metrics);
}

void FetchLoader::didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError& error)
{
    m_client->didFail(error);
}

} // namespace WebCore
