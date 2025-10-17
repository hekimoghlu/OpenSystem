/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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
#include "CachedScriptFetcher.h"

#include "CachedResourceLoader.h"
#include "CachedScript.h"
#include "ContentSecurityPolicy.h"
#include "CrossOriginAccessControl.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "Settings.h"
#include "WorkerOrWorkletGlobalScope.h"

namespace WebCore {

Ref<CachedScriptFetcher> CachedScriptFetcher::create(const AtomString& charset)
{
    return adoptRef(*new CachedScriptFetcher(charset));
}

CachedResourceHandle<CachedScript> CachedScriptFetcher::requestModuleScript(Document& document, const URL& sourceURL, String&& integrity, std::optional<ServiceWorkersMode> serviceWorkersMode) const
{
    return requestScriptWithCache(document, sourceURL, String { }, WTFMove(integrity), { }, serviceWorkersMode);
}

CachedResourceHandle<CachedScript> CachedScriptFetcher::requestScriptWithCache(Document& document, const URL& sourceURL, const String& crossOriginMode, String&& integrity, std::optional<ResourceLoadPriority> resourceLoadPriority, std::optional<ServiceWorkersMode> serviceWorkersMode) const
{
    if (!document.settings().isScriptEnabled())
        return nullptr;

    ASSERT(document.contentSecurityPolicy());
    bool hasKnownNonce = document.checkedContentSecurityPolicy()->allowScriptWithNonce(m_nonce, m_isInUserAgentShadowTree);
    ResourceLoaderOptions options = CachedResourceLoader::defaultCachedResourceOptions();
    options.contentSecurityPolicyImposition = hasKnownNonce ? ContentSecurityPolicyImposition::SkipPolicyCheck : ContentSecurityPolicyImposition::DoPolicyCheck;
    options.sameOriginDataURLFlag = SameOriginDataURLFlag::Set;
    options.serviceWorkersMode = serviceWorkersMode.value_or(ServiceWorkersMode::All);
    options.integrity = WTFMove(integrity);
    options.referrerPolicy = m_referrerPolicy;
    options.fetchPriority = m_fetchPriority;
    options.nonce = m_nonce;

    auto request = createPotentialAccessControlRequest(sourceURL, WTFMove(options), document, crossOriginMode);
    request.upgradeInsecureRequestIfNeeded(document);
    request.setCharset(m_charset);
    request.setPriority(WTFMove(resourceLoadPriority));
    if (!m_initiatorType.isNull())
        request.setInitiatorType(m_initiatorType);

    return document.protectedCachedResourceLoader()->requestScript(WTFMove(request)).value_or(nullptr);
}

}
