/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
#include "JSGlobalObjectAuditAgent.h"

#include "InjectedScript.h"
#include "InjectedScriptManager.h"
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(JSGlobalObjectAuditAgent);

JSGlobalObjectAuditAgent::JSGlobalObjectAuditAgent(JSAgentContext& context)
    : InspectorAuditAgent(context)
    , m_globalObject(context.inspectedGlobalObject)
{
}

JSGlobalObjectAuditAgent::~JSGlobalObjectAuditAgent() = default;

InjectedScript JSGlobalObjectAuditAgent::injectedScriptForEval(Protocol::ErrorString& errorString, std::optional<Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    if (executionContextId) {
        errorString = "executionContextId is not supported for JSContexts as there is only one execution context"_s;
        return InjectedScript();
    }

    InjectedScript injectedScript = injectedScriptManager().injectedScriptFor(&m_globalObject);
    if (injectedScript.hasNoValue())
        errorString = "Internal error: main world execution context not found"_s;

    return injectedScript;
}

} // namespace Inspector
