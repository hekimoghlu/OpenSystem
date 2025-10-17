/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#include "JSGlobalObjectDebuggerAgent.h"

#include "ConsoleMessage.h"
#include "InjectedScriptManager.h"
#include "InspectorConsoleAgent.h"
#include "JSGlobalObjectDebugger.h"
#include "ScriptCallStackFactory.h"
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(JSGlobalObjectDebuggerAgent);

JSGlobalObjectDebuggerAgent::JSGlobalObjectDebuggerAgent(JSAgentContext& context, InspectorConsoleAgent* consoleAgent)
    : InspectorDebuggerAgent(context)
    , m_consoleAgent(consoleAgent)
{
}

JSGlobalObjectDebuggerAgent::~JSGlobalObjectDebuggerAgent() = default;

InjectedScript JSGlobalObjectDebuggerAgent::injectedScriptForEval(Protocol::ErrorString& errorString, std::optional<Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    if (executionContextId) {
        errorString = "executionContextId is not supported for JSContexts as there is only one execution context"_s;
        return InjectedScript();
    }

    JSGlobalObject& globalObject = static_cast<JSGlobalObjectDebugger&>(debugger()).globalObject();
    return injectedScriptManager().injectedScriptFor(&globalObject);
}

void JSGlobalObjectDebuggerAgent::breakpointActionLog(JSC::JSGlobalObject* globalObject, const String& message)
{
    m_consoleAgent->addMessageToConsole(makeUnique<ConsoleMessage>(MessageSource::JS, MessageType::Log, MessageLevel::Log, message, createScriptCallStack(globalObject), 0));
}

} // namespace Inspector
