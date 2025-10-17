/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
#include "Worklet.h"

#include "ContentSecurityPolicy.h"
#include "Document.h"
#include "JSDOMPromiseDeferred.h"
#include "Page.h"
#include "ScriptSourceCode.h"
#include "SecurityOrigin.h"
#include "WorkerRunLoop.h"
#include "WorkletGlobalScope.h"
#include "WorkletGlobalScopeProxy.h"
#include "WorkletPendingTasks.h"
#include <JavaScriptCore/IdentifiersFactory.h>
#include <wtf/CrossThreadCopier.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Worklet);

Worklet::Worklet(Document& document)
    : ActiveDOMObject(&document)
    , m_identifier(makeString("worklet:"_s, Inspector::IdentifiersFactory::createIdentifier()))
{
}

Worklet::~Worklet() = default;

Document* Worklet::document()
{
    return downcast<Document>(scriptExecutionContext());
}

// https://www.w3.org/TR/worklets-1/#dom-worklet-addmodule
void Worklet::addModule(const String& moduleURLString, WorkletOptions&& options, DOMPromiseDeferred<void>&& promise)
{
    auto* document = this->document();
    if (!document || !document->page()) {
        promise.reject(Exception { ExceptionCode::InvalidStateError, "This frame is detached"_s });
        return;
    }

    URL moduleURL = document->completeURL(moduleURLString);
    if (!moduleURL.isValid()) {
        promise.reject(Exception { ExceptionCode::SyntaxError, "Module URL is invalid"_s });
        return;
    }

    if (!document->checkedContentSecurityPolicy()->allowScriptFromSource(moduleURL)) {
        promise.reject(Exception { ExceptionCode::SecurityError, "Not allowed by CSP"_s });
        return;
    }

    if (m_proxies.isEmpty())
        m_proxies.appendVector(createGlobalScopes());

    auto pendingTasks = WorkletPendingTasks::create(*this, WTFMove(promise), m_proxies.size());
    m_pendingTasksSet.add(pendingTasks.copyRef());

    for (auto& proxy : m_proxies) {
        proxy->postTaskForModeToWorkletGlobalScope([pendingTasks = pendingTasks.copyRef(), moduleURL = moduleURL.isolatedCopy(), credentials = options.credentials, pendingActivity = makePendingActivity(*this)](ScriptExecutionContext& context) mutable {
            downcast<WorkletGlobalScope>(context).fetchAndInvokeScript(moduleURL, credentials, [pendingTasks = WTFMove(pendingTasks), pendingActivity = WTFMove(pendingActivity)](std::optional<Exception>&& exception) mutable {
                callOnMainThread([pendingTasks = WTFMove(pendingTasks), exception = crossThreadCopy(WTFMove(exception)), pendingActivity = WTFMove(pendingActivity)]() mutable {
                    if (exception)
                        pendingTasks->abort(WTFMove(*exception));
                    else
                        pendingTasks->decrementCounter();
                });
            });
        }, WorkerRunLoop::defaultMode());
    }
}

void Worklet::finishPendingTasks(WorkletPendingTasks& tasks)
{
    ASSERT(isMainThread());
    ASSERT(m_pendingTasksSet.contains(&tasks));

    m_pendingTasksSet.remove(&tasks);
}

} // namespace WebCore
