/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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
#include "CachedModuleScriptLoader.h"

#include "CachedScript.h"
#include "CachedScriptFetcher.h"
#include "DOMWrapperWorld.h"
#include "JSDOMBinding.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalFrame.h"
#include "ModuleFetchParameters.h"
#include "ResourceLoaderOptions.h"
#include "ScriptController.h"
#include "ScriptModuleLoader.h"
#include "ScriptSourceCode.h"

namespace WebCore {

Ref<CachedModuleScriptLoader> CachedModuleScriptLoader::create(ModuleScriptLoaderClient& client, DeferredPromise& promise, CachedScriptFetcher& scriptFetcher, RefPtr<JSC::ScriptFetchParameters>&& parameters)
{
    return adoptRef(*new CachedModuleScriptLoader(client, promise, scriptFetcher, WTFMove(parameters)));
}

CachedModuleScriptLoader::CachedModuleScriptLoader(ModuleScriptLoaderClient& client, DeferredPromise& promise, CachedScriptFetcher& scriptFetcher, RefPtr<JSC::ScriptFetchParameters>&& parameters)
    : ModuleScriptLoader(client, promise, scriptFetcher, WTFMove(parameters))
{
}

CachedModuleScriptLoader::~CachedModuleScriptLoader()
{
    if (m_cachedScript) {
        m_cachedScript->removeClient(*this);
        m_cachedScript = nullptr;
    }
}

bool CachedModuleScriptLoader::load(Document& document, URL&& sourceURL, std::optional<ServiceWorkersMode> serviceWorkersMode)
{
    ASSERT(m_promise);
    ASSERT(!m_cachedScript);
    String integrity = m_parameters ? m_parameters->integrity() : String { };
    m_cachedScript = scriptFetcher().requestModuleScript(document, sourceURL, WTFMove(integrity), serviceWorkersMode);
    if (!m_cachedScript)
        return false;
    m_sourceURL = WTFMove(sourceURL);

    // If the content is already cached, this immediately calls notifyFinished.
    m_cachedScript->addClient(*this);
    return true;
}

void CachedModuleScriptLoader::notifyFinished(CachedResource& resource, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    ASSERT_UNUSED(resource, &resource == m_cachedScript);
    ASSERT(m_cachedScript);
    ASSERT(m_promise);

    Ref<CachedModuleScriptLoader> protectedThis(*this);
    if (m_client)
        m_client->notifyFinished(*this, WTFMove(m_sourceURL), m_promise.releaseNonNull());

    // Remove the client after calling notifyFinished to keep the data buffer in
    // CachedResource alive while notifyFinished processes the resource.
    m_cachedScript->removeClient(*this);
    m_cachedScript = nullptr;
}

}
