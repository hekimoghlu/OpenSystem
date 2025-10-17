/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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
#include "InspectorAuditAgent.h"

#include "Debugger.h"
#include "InjectedScript.h"
#include "JSLock.h"
#include "ObjectConstructor.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorAuditAgent);

InspectorAuditAgent::InspectorAuditAgent(AgentContext& context)
    : InspectorAgentBase("Audit"_s)
    , m_backendDispatcher(AuditBackendDispatcher::create(context.backendDispatcher, this))
    , m_injectedScriptManager(context.injectedScriptManager)
    , m_debugger(*context.environment.debugger())
{
}

InspectorAuditAgent::~InspectorAuditAgent() = default;

void InspectorAuditAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
}

void InspectorAuditAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
}

Protocol::ErrorStringOr<void> InspectorAuditAgent::setup(std::optional<Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    Protocol::ErrorString errorString;

    if (hasActiveAudit())
        return makeUnexpected("Must call teardown before calling setup again"_s);

    InjectedScript injectedScript = injectedScriptForEval(errorString, WTFMove(executionContextId));
    if (injectedScript.hasNoValue())
        return makeUnexpected(errorString);

    JSC::JSGlobalObject* globalObject = injectedScript.globalObject();
    if (!globalObject)
        return makeUnexpected("Missing execution state of injected script for given executionContextId"_s);

    VM& vm = globalObject->vm();

    JSC::JSLockHolder lock(globalObject);

    m_injectedWebInspectorAuditValue.set(vm, constructEmptyObject(globalObject));
    if (!m_injectedWebInspectorAuditValue)
        return makeUnexpected("Unable to construct injected WebInspectorAudit object."_s);

    populateAuditObject(globalObject, m_injectedWebInspectorAuditValue);

    return { };
}

Protocol::ErrorStringOr<std::tuple<Ref<Protocol::Runtime::RemoteObject>, std::optional<bool> /* wasThrown */>> InspectorAuditAgent::run(const String& test, std::optional<Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    Protocol::ErrorString errorString;

    InjectedScript injectedScript = injectedScriptForEval(errorString, WTFMove(executionContextId));
    if (injectedScript.hasNoValue())
        return makeUnexpected(errorString);

    auto functionString = makeString("(function(WebInspectorAudit) { \"use strict\"; return eval(`("_s, makeStringByReplacingAll(test, '`', "\\`"_s), ")`)(WebInspectorAudit); })"_s);

    InjectedScript::ExecuteOptions options;
    options.objectGroup = "audit"_s;
    if (m_injectedWebInspectorAuditValue)
        options.args = { m_injectedWebInspectorAuditValue.get() };

    RefPtr<Protocol::Runtime::RemoteObject> result;
    std::optional<bool> wasThrown;
    std::optional<int> savedResultIndex;

    JSC::Debugger::TemporarilyDisableExceptionBreakpoints temporarilyDisableExceptionBreakpoints(m_debugger);
    temporarilyDisableExceptionBreakpoints.replace();

    muteConsole();

    injectedScript.execute(errorString, functionString, WTFMove(options), result, wasThrown, savedResultIndex);

    unmuteConsole();

    if (!result)
        return makeUnexpected(errorString);

    return { { result.releaseNonNull(), WTFMove(wasThrown) } };
}

Protocol::ErrorStringOr<void> InspectorAuditAgent::teardown()
{
    if (!hasActiveAudit())
        return makeUnexpected("Must call setup before calling teardown"_s);

    m_injectedWebInspectorAuditValue.clear();

    return { };
}

bool InspectorAuditAgent::hasActiveAudit() const
{
    return !!m_injectedWebInspectorAuditValue;
}

void InspectorAuditAgent::populateAuditObject(JSC::JSGlobalObject* globalObject, JSC::Strong<JSC::JSObject>& auditObject)
{
    ASSERT(globalObject);
    if (!globalObject)
        return;

    JSC::VM& vm = globalObject->vm();
    JSC::JSLockHolder lock(vm);

    auditObject->putDirect(vm, JSC::Identifier::fromString(vm, "Version"_s), JSC::JSValue(Protocol::Audit::VERSION));
}

} // namespace Inspector
