/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
#include "HTMLResourcePreloader.h"

#include "CachedResourceLoader.h"
#include "CrossOriginAccessControl.h"
#include "DefaultResourceLoadPriority.h"
#include "DocumentInlines.h"
#include "MediaQueryEvaluator.h"
#include "MediaQueryParser.h"
#include "NodeRenderStyle.h"
#include "RenderView.h"
#include "ScriptElementCachedScriptFetcher.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PreloadRequest);
WTF_MAKE_TZONE_ALLOCATED_IMPL(HTMLResourcePreloader);

URL PreloadRequest::completeURL(Document& document)
{
    return document.completeURL(m_resourceURL, m_baseURL.isEmpty() ? document.baseURL() : m_baseURL);
}

CachedResourceRequest PreloadRequest::resourceRequest(Document& document)
{
    ASSERT(isMainThread());

    bool skipContentSecurityPolicyCheck = false;
    if (m_resourceType == CachedResource::Type::Script)
        skipContentSecurityPolicyCheck = document.checkedContentSecurityPolicy()->allowScriptWithNonce(m_nonceAttribute);
    else if (m_resourceType == CachedResource::Type::CSSStyleSheet)
        skipContentSecurityPolicyCheck = document.checkedContentSecurityPolicy()->allowStyleWithNonce(m_nonceAttribute);

    ResourceLoaderOptions options = CachedResourceLoader::defaultCachedResourceOptions();
    if (skipContentSecurityPolicyCheck)
        options.contentSecurityPolicyImposition = ContentSecurityPolicyImposition::SkipPolicyCheck;

    String crossOriginMode = m_crossOriginMode;
    if (m_scriptType == ScriptType::Module) {
        if (crossOriginMode.isNull())
            crossOriginMode = ScriptElementCachedScriptFetcher::defaultCrossOriginModeForModule;
    }
    if (m_resourceType == CachedResource::Type::Script || m_resourceType == CachedResource::Type::ImageResource)
        options.referrerPolicy = m_referrerPolicy;
    options.fetchPriority = m_fetchPriority;
    auto request = createPotentialAccessControlRequest(completeURL(document), WTFMove(options), document, crossOriginMode);
    request.setInitiatorType(m_initiatorType);

    if (m_scriptIsAsync && m_resourceType == CachedResource::Type::Script && m_scriptType == ScriptType::Classic)
        request.setPriority(DefaultResourceLoadPriority::asyncScript);

    return request;
}

void HTMLResourcePreloader::preload(PreloadRequestStream requests)
{
    for (auto& request : requests)
        preload(WTFMove(request));
}

void HTMLResourcePreloader::preload(std::unique_ptr<PreloadRequest> preload)
{
    Ref document = protectedDocument();
    ASSERT(document->frame());
    ASSERT(document->renderView());

    auto queries = MQ::MediaQueryParser::parse(preload->media(), { document.get() });
    if (!MQ::MediaQueryEvaluator { screenAtom(), document, document->renderStyle() }.evaluate(queries))
        return;

    document->protectedCachedResourceLoader()->preload(preload->resourceType(), preload->resourceRequest(document));
}

}
