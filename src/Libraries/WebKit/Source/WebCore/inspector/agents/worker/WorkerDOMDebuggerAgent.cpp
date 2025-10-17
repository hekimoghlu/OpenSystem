/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#include "WorkerDOMDebuggerAgent.h"

namespace WebCore {

using namespace Inspector;

WorkerDOMDebuggerAgent::WorkerDOMDebuggerAgent(WorkerAgentContext& context, InspectorDebuggerAgent* debuggerAgent)
    : InspectorDOMDebuggerAgent(context, debuggerAgent)
{
}

WorkerDOMDebuggerAgent::~WorkerDOMDebuggerAgent() = default;

Inspector::Protocol::ErrorStringOr<void> WorkerDOMDebuggerAgent::setDOMBreakpoint(Inspector::Protocol::DOM::NodeId, Inspector::Protocol::DOMDebugger::DOMBreakpointType, RefPtr<JSON::Object>&& /* options */)
{
    return makeUnexpected("Not supported"_s);
}

Inspector::Protocol::ErrorStringOr<void> WorkerDOMDebuggerAgent::removeDOMBreakpoint(Inspector::Protocol::DOM::NodeId, Inspector::Protocol::DOMDebugger::DOMBreakpointType)
{
    return makeUnexpected("Not supported"_s);
}

bool WorkerDOMDebuggerAgent::setAnimationFrameBreakpoint(Inspector::Protocol::ErrorString& errorString, RefPtr<JSC::Breakpoint>&&)
{
    errorString = "Not supported"_s;
    return false;
}

} // namespace WebCore
