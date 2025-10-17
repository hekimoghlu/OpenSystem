/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#include "SharedWorkerScriptLoader.h"

#include "EventNames.h"
#include "InspectorInstrumentation.h"
#include "SharedWorker.h"
#include "WorkerFetchResult.h"
#include "WorkerInitializationData.h"
#include "WorkerRunLoop.h"
#include "WorkerScriptLoader.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SharedWorkerScriptLoader);

SharedWorkerScriptLoader::SharedWorkerScriptLoader(URL&& url, SharedWorker& worker, WorkerOptions&& options)
    : m_options(WTFMove(options))
    , m_worker(worker)
    , m_loader(WorkerScriptLoader::create())
    , m_url(WTFMove(url))
{
}

void SharedWorkerScriptLoader::load(CompletionHandler<void(WorkerFetchResult&&, WorkerInitializationData&&)>&& completionHandler)
{
    ASSERT(!m_completionHandler);
    m_completionHandler = WTFMove(completionHandler);

    auto source = m_options.type == WorkerType::Module ? WorkerScriptLoader::Source::ModuleScript : WorkerScriptLoader::Source::ClassicWorkerScript;
    m_loader->loadAsynchronously(*m_worker->scriptExecutionContext(), ResourceRequest(m_url), source, m_worker->workerFetchOptions(m_options, FetchOptions::Destination::Sharedworker), ContentSecurityPolicyEnforcement::EnforceWorkerSrcDirective, ServiceWorkersMode::All, *this, WorkerRunLoop::defaultMode(), ScriptExecutionContextIdentifier::generate());
}

void SharedWorkerScriptLoader::didReceiveResponse(ScriptExecutionContextIdentifier mainContextIdentifier, std::optional<ResourceLoaderIdentifier> identifier, const ResourceResponse&)
{
    if (UNLIKELY(InspectorInstrumentation::hasFrontends())) {
        ScriptExecutionContext::ensureOnContextThread(mainContextIdentifier, [identifier] (auto& mainContext) {
            InspectorInstrumentation::didReceiveScriptResponse(mainContext, *identifier);
        });
    }
}

void SharedWorkerScriptLoader::notifyFinished(std::optional<ScriptExecutionContextIdentifier> mainContextIdentifier)
{
    auto* scriptExecutionContext = m_worker->scriptExecutionContext();

    if (UNLIKELY(InspectorInstrumentation::hasFrontends()) && scriptExecutionContext && !m_loader->failed()) {
        ScriptExecutionContext::ensureOnContextThread(*mainContextIdentifier, [identifier = m_loader->identifier(), script = m_loader->script().isolatedCopy()] (auto& mainContext) {
            InspectorInstrumentation::scriptImported(mainContext, identifier, script.toString());
        });
    }

    auto fetchResult = m_loader->fetchResult();
    if (fetchResult.referrerPolicy.isNull() && scriptExecutionContext)
        fetchResult.referrerPolicy = referrerPolicyToString(scriptExecutionContext->referrerPolicy());
    m_completionHandler(WTFMove(fetchResult), WorkerInitializationData {
        m_loader->takeServiceWorkerData(),
        m_loader->clientIdentifier(),
        m_loader->advancedPrivacyProtections(),
        m_loader->userAgentForSharedWorker()
    }); // deletes this.
}

} // namespace WebCore
