/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
#pragma once

#include "InspectorDebuggerAgent.h"
#include <wtf/TZoneMalloc.h>

namespace Inspector {

class InspectorConsoleAgent;

class JSGlobalObjectDebuggerAgent final : public InspectorDebuggerAgent {
    WTF_MAKE_NONCOPYABLE(JSGlobalObjectDebuggerAgent);
    WTF_MAKE_TZONE_ALLOCATED(JSGlobalObjectDebuggerAgent);
public:
    JSGlobalObjectDebuggerAgent(JSAgentContext&, InspectorConsoleAgent*);
    ~JSGlobalObjectDebuggerAgent() final;

    // JSC::Debugger::Observer
    void breakpointActionLog(JSC::JSGlobalObject*, const String& data) final;

private:
    InjectedScript injectedScriptForEval(Protocol::ErrorString&, std::optional<Protocol::Runtime::ExecutionContextId>&&) final;

    // NOTE: JavaScript inspector does not yet need to mute a console because no messages
    // are sent to the console outside of the API boundary or console object.
    void muteConsole() final { }
    void unmuteConsole() final { }

    InspectorConsoleAgent* m_consoleAgent { nullptr };
};

} // namespace Inspector
