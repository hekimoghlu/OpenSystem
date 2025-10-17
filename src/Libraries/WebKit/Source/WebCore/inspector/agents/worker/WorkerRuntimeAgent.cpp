/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
#include "WorkerRuntimeAgent.h"

#include "JSDOMGlobalObject.h"
#include "WorkerOrWorkletGlobalScope.h"
#include "WorkerOrWorkletScriptController.h"
#include <JavaScriptCore/InjectedScript.h>
#include <JavaScriptCore/InjectedScriptManager.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WorkerRuntimeAgent);

WorkerRuntimeAgent::WorkerRuntimeAgent(WorkerAgentContext& context)
    : InspectorRuntimeAgent(context)
    , m_backendDispatcher(RuntimeBackendDispatcher::create(context.backendDispatcher, this))
    , m_globalScope(context.globalScope)
{
    ASSERT(context.globalScope->isContextThread());
}

WorkerRuntimeAgent::~WorkerRuntimeAgent() = default;

InjectedScript WorkerRuntimeAgent::injectedScriptForEval(Inspector::Protocol::ErrorString& errorString, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    if (executionContextId) {
        errorString = "executionContextId is not supported for workers as there is only one execution context"_s;
        return InjectedScript();
    }

    // FIXME: What guarantees m_globalScope.script() is non-null?
    // FIXME: What guarantees globalScopeWrapper() is non-null?
    return injectedScriptManager().injectedScriptFor(m_globalScope.script()->globalScopeWrapper());
}

} // namespace WebCore
