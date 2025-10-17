/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 30, 2025.
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
#include "WorkerFontLoadRequest.h"

#include "Font.h"
#include "FontCreationContext.h"
#include "FontCustomPlatformData.h"
#include "FontSelectionAlgorithm.h"
#include "ResourceLoaderOptions.h"
#include "ServiceWorker.h"
#include "WOFFFileFormat.h"
#include "WorkerGlobalScope.h"
#include "WorkerThreadableLoader.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerFontLoadRequest);

WorkerFontLoadRequest::WorkerFontLoadRequest(URL&& url, LoadedFromOpaqueSource loadedFromOpaqueSource)
    : m_url(WTFMove(url))
    , m_loadedFromOpaqueSource(loadedFromOpaqueSource)
{
}

void WorkerFontLoadRequest::load(WorkerGlobalScope& workerGlobalScope)
{
    m_context = workerGlobalScope;

    ResourceRequest request { m_url };
    ASSERT(request.httpMethod() == "GET"_s);

    FetchOptions fetchOptions;
    fetchOptions.mode = FetchOptions::Mode::SameOrigin;
    fetchOptions.credentials = workerGlobalScope.credentials();
    fetchOptions.cache = FetchOptions::Cache::Default;
    fetchOptions.redirect = FetchOptions::Redirect::Follow;
    fetchOptions.destination = FetchOptions::Destination::Worker;

    ThreadableLoaderOptions options { WTFMove(fetchOptions) };
    options.sendLoadCallbacks = SendCallbackPolicy::SendCallbacks;
    options.contentSecurityPolicyEnforcement = m_context->shouldBypassMainWorldContentSecurityPolicy() ? ContentSecurityPolicyEnforcement::DoNotEnforce : ContentSecurityPolicyEnforcement::EnforceWorkerSrcDirective;
    options.loadedFromOpaqueSource = m_loadedFromOpaqueSource;
    options.sameOriginDataURLFlag = SameOriginDataURLFlag::Set;

    options.serviceWorkersMode = ServiceWorkersMode::All;
    if (auto* activeServiceWorker = workerGlobalScope.activeServiceWorker())
        options.serviceWorkerRegistrationIdentifier = activeServiceWorker->registrationIdentifier();

    WorkerThreadableLoader::loadResourceSynchronously(workerGlobalScope, WTFMove(request), *this, options);
}

bool WorkerFontLoadRequest::ensureCustomFontData()
{
    if (!m_fontCustomPlatformData && !m_errorOccurred && !m_isLoading) {
        RefPtr<SharedBuffer> contiguousData;
        if (m_data)
            contiguousData = m_data.takeAsContiguous();
        convertWOFFToSfntIfNecessary(contiguousData);
        if (contiguousData) {
            m_fontCustomPlatformData = FontCustomPlatformData::create(*contiguousData, m_url.fragmentIdentifier().toString());
            m_data = WTFMove(contiguousData);
            if (!m_fontCustomPlatformData)
                m_errorOccurred = true;
        }
    }

    return m_fontCustomPlatformData.get();
}

RefPtr<Font> WorkerFontLoadRequest::createFont(const FontDescription& fontDescription, bool syntheticBold, bool syntheticItalic, const FontCreationContext& fontCreationContext)
{
    ASSERT(m_fontCustomPlatformData);
    ASSERT(m_context);
    return Font::create(m_fontCustomPlatformData->fontPlatformData(fontDescription, syntheticBold, syntheticItalic, fontCreationContext), Font::Origin::Remote);
}

void WorkerFontLoadRequest::setClient(FontLoadRequestClient* client)
{
    m_fontLoadRequestClient = client;

    if (m_notifyOnClientSet) {
        m_notifyOnClientSet = false;
        m_fontLoadRequestClient->fontLoaded(*this);
    }
}

void WorkerFontLoadRequest::didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse& response)
{
    if (response.httpStatusCode() / 100 != 2 && response.httpStatusCode())
        m_errorOccurred = true;
}

void WorkerFontLoadRequest::didReceiveData(const SharedBuffer& buffer)
{
    if (m_errorOccurred)
        return;

    m_data.append(buffer);
}

void WorkerFontLoadRequest::didFinishLoading(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const NetworkLoadMetrics&)
{
    m_isLoading = false;

    if (!m_errorOccurred) {
        if (m_fontLoadRequestClient)
            m_fontLoadRequestClient->fontLoaded(*this);
        else
            m_notifyOnClientSet = true;
    }
}

void WorkerFontLoadRequest::didFail(std::optional<ScriptExecutionContextIdentifier>, const ResourceError&)
{
    m_errorOccurred = true;
    if (m_fontLoadRequestClient)
        m_fontLoadRequestClient->fontLoaded(*this);
}

} // namespace WebCore
