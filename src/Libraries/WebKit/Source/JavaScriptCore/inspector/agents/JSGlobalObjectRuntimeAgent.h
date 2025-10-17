/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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

#include "InspectorFrontendDispatchers.h"
#include "InspectorRuntimeAgent.h"
#include <wtf/TZoneMalloc.h>

namespace JSC {
class JSGlobalObject;
}

namespace Inspector {

class JSGlobalObjectRuntimeAgent final : public InspectorRuntimeAgent {
    WTF_MAKE_NONCOPYABLE(JSGlobalObjectRuntimeAgent);
    WTF_MAKE_TZONE_ALLOCATED(JSGlobalObjectRuntimeAgent);
public:
    JSGlobalObjectRuntimeAgent(JSAgentContext&);
    ~JSGlobalObjectRuntimeAgent() final;

private:
    InjectedScript injectedScriptForEval(Protocol::ErrorString&, std::optional<Protocol::Runtime::ExecutionContextId>&&) final;

    // NOTE: JavaScript inspector does not yet need to mute a console because no messages
    // are sent to the console outside of the API boundary or console object.
    void muteConsole() final { }
    void unmuteConsole() final { }

    std::unique_ptr<RuntimeFrontendDispatcher> m_frontendDispatcher;
    RefPtr<RuntimeBackendDispatcher> m_backendDispatcher;
    JSC::JSGlobalObject& m_globalObject;
};

} // namespace Inspector
