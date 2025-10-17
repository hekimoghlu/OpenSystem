/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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
#include "WorkerModuleScriptLoader.h"

#include "CachedScriptFetcher.h"
#include "ContentSecurityPolicy.h"
#include "DOMWrapperWorld.h"
#include "JSDOMBinding.h"
#include "JSDOMPromiseDeferred.h"
#include "LocalFrame.h"
#include "ModuleFetchParameters.h"
#include "ResourceLoaderOptions.h"
#include "ScriptController.h"
#include "ScriptModuleLoader.h"
#include "ScriptSourceCode.h"
#include "ServiceWorkerGlobalScope.h"
#include "WorkerScriptFetcher.h"
#include "WorkerScriptLoader.h"

namespace WebCore {

Ref<WorkerModuleScriptLoader> WorkerModuleScriptLoader::create(ModuleScriptLoaderClient& client, DeferredPromise& promise, WorkerScriptFetcher& scriptFetcher, RefPtr<JSC::ScriptFetchParameters>&& parameters)
{
    return adoptRef(*new WorkerModuleScriptLoader(client, promise, scriptFetcher, WTFMove(parameters)));
}

WorkerModuleScriptLoader::WorkerModuleScriptLoader(ModuleScriptLoaderClient& client, DeferredPromise& promise, WorkerScriptFetcher& scriptFetcher, RefPtr<JSC::ScriptFetchParameters>&& parameters)
    : ModuleScriptLoader(client, promise, scriptFetcher, WTFMove(parameters))
    , m_scriptLoader(WorkerScriptLoader::create())
{
}

WorkerModuleScriptLoader::~WorkerModuleScriptLoader()
{
    protectedScriptLoader()->cancel();
}

void WorkerModuleScriptLoader::load(ScriptExecutionContext& context, URL&& sourceURL)
{
    m_sourceURL = WTFMove(sourceURL);

    if (auto* globalScope = dynamicDowncast<ServiceWorkerGlobalScope>(context)) {
        if (auto* scriptResource = globalScope->scriptResource(m_sourceURL)) {
            m_script = scriptResource->script;
            m_responseURL = scriptResource->responseURL;
            m_responseMIMEType = scriptResource->mimeType;
            m_retrievedFromServiceWorkerCache = true;
            notifyClientFinished();
            return;
        }
    }

    ResourceRequest request { m_sourceURL };

    FetchOptions fetchOptions;
    fetchOptions.mode = FetchOptions::Mode::Cors;
    fetchOptions.cache = FetchOptions::Cache::Default;
    fetchOptions.redirect = FetchOptions::Redirect::Follow;
    fetchOptions.credentials = static_cast<WorkerScriptFetcher&>(scriptFetcher()).credentials();
    fetchOptions.destination = static_cast<WorkerScriptFetcher&>(scriptFetcher()).destination();
    fetchOptions.referrerPolicy = static_cast<WorkerScriptFetcher&>(scriptFetcher()).referrerPolicy();

    bool cspCheckFailed = false;
    ContentSecurityPolicyEnforcement contentSecurityPolicyEnforcement = ContentSecurityPolicyEnforcement::DoNotEnforce;
    if (!context.shouldBypassMainWorldContentSecurityPolicy()) {
        CheckedPtr contentSecurityPolicy = context.contentSecurityPolicy();
        if (fetchOptions.destination == FetchOptions::Destination::Script) {
            cspCheckFailed = contentSecurityPolicy && !contentSecurityPolicy->allowScriptFromSource(m_sourceURL);
            contentSecurityPolicyEnforcement = ContentSecurityPolicyEnforcement::EnforceScriptSrcDirective;
        } else {
            cspCheckFailed = contentSecurityPolicy && !contentSecurityPolicy->allowWorkerFromSource(m_sourceURL);
            contentSecurityPolicyEnforcement = ContentSecurityPolicyEnforcement::EnforceWorkerSrcDirective;
        }
    }

    if (cspCheckFailed) {
        // FIXME: Always get the `ScriptExecutionContextIdentifier` of the `Document`.
        std::optional<ScriptExecutionContextIdentifier> mainContext;
        if (auto* document = dynamicDowncast<Document>(context))
            mainContext = document->identifier();
        protectedScriptLoader()->notifyError(mainContext);
        ASSERT(!m_failed);
        notifyFinished(mainContext);
        ASSERT(m_failed);
        return;
    }

    // https://html.spec.whatwg.org/multipage/webappapis.html#fetch-a-single-module-script
    // If destination is "worker" or "sharedworker" and the top-level module fetch flag is set, then set request's mode to "same-origin".
    if (fetchOptions.destination == FetchOptions::Destination::Worker || fetchOptions.destination == FetchOptions::Destination::Serviceworker) {
        if (parameters() && parameters()->isTopLevelModule())
            fetchOptions.mode = FetchOptions::Mode::SameOrigin;
    }

    protectedScriptLoader()->loadAsynchronously(context, WTFMove(request), WorkerScriptLoader::Source::ModuleScript, WTFMove(fetchOptions), contentSecurityPolicyEnforcement, ServiceWorkersMode::All, *this, taskMode());
}

Ref<WorkerScriptLoader> WorkerModuleScriptLoader::protectedScriptLoader()
{
    return m_scriptLoader;
}

ReferrerPolicy WorkerModuleScriptLoader::referrerPolicy()
{
    if (auto policy = parseReferrerPolicy(m_scriptLoader->referrerPolicy(), ReferrerPolicySource::HTTPHeader))
        return *policy;
    return ReferrerPolicy::EmptyString;
}

void WorkerModuleScriptLoader::notifyFinished(std::optional<ScriptExecutionContextIdentifier>)
{
    ASSERT(m_promise);

    if (m_scriptLoader->failed())
        m_failed = true;
    else {
        m_script = m_scriptLoader->script();
        m_responseURL = m_scriptLoader->responseURL();
        m_responseMIMEType = m_scriptLoader->responseMIMEType();
    }

    notifyClientFinished();
}

void WorkerModuleScriptLoader::notifyClientFinished()
{
    Ref protectedThis { *this };

    if (m_client)
        m_client->notifyFinished(*this, WTFMove(m_sourceURL), m_promise.releaseNonNull());
}

String WorkerModuleScriptLoader::taskMode()
{
    return "loadModulesInWorkerOrWorkletMode"_s;
}

}
